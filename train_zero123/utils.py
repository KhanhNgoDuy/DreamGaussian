from diffusers import DDIMScheduler
import torchvision.transforms.functional as TF

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('./')

from zero123 import Zero123Pipeline


class Zero123(nn.Module):
    def __init__(self, cfg, fp16=True, t_range=[0.02, 0.98], model_key="ashawkey/zero123-xl-diffusers", trained_module='attentions'):
        super().__init__()

        self.cfg = cfg
        self.device = cfg.device
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32

        assert self.fp16, 'Only zero123 fp16 is supported for now.'

        self.trained_module = cfg.trained_module or 'attentions'
        assert self.trained_module in ['attentions', 'mid_block', 'unet']

        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.accumulate_grad_batches = cfg.accumulate_grad_batches

        self.pipe = Zero123Pipeline.from_pretrained(
            model_key,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)

        # stable-zero123 has a different camera embedding
        self.use_stable_zero123 = 'stable' in model_key

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.clip_camera_projection = self.pipe.clip_camera_projection
        self.setup_model()

        self.pipe.set_progress_bar_config(disable=True)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps


        self.min_step = 1
        self.max_step = 5
        # self.min_step = int(self.num_train_timesteps * t_range[0])
        # self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        # print(self.num_train_timesteps)
        # exit()

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor in [0, 1]
        x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        x_pil = [TF.to_pil_image(image) for image in x]
        x_clip = self.pipe.feature_extractor(images=x_pil, return_tensors="pt").pixel_values.to(device=self.device, dtype=self.dtype)
        c = self.pipe.image_encoder(x_clip).image_embeds
        v = self.encode_imgs(x.to(self.dtype)) / self.vae.config.scaling_factor

        return [c, v]
    
    def get_cam_embeddings(self, elevation, azimuth, radius, default_elevation=0):
        T = np.stack([np.deg2rad(elevation), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), radius], axis=-1)
        T = torch.from_numpy(T).unsqueeze(1).to(dtype=self.dtype, device=self.device) # [8, 1, 4]
        return T
    
    def forward(self, pred_rgb, cond_rgb, data, step_ratio=None, guidance_scale=5, as_latent=False, default_elevation=0):
        # print(data.shape)
        elevation, azimuth, radius = -data[:, 0].item(), -data[:, 1].item(), -data[:, 2].item()
        batch_size = pred_rgb.shape[0]

        pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_256.to(self.dtype))

        if step_ratio is not None:
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        with torch.enable_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)

            T = self.get_cam_embeddings([elevation], [azimuth], [radius], default_elevation)      # [1, 1, 4]
            embeddings = self.get_img_embeds(pred_rgb_256)

            cc_emb = torch.cat([embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)    # [1, 768] --> [1, 1, 768]  +   [1, 1, 4] --> [1, 1, 772]
            cc_emb = self.pipe.clip_camera_projection(cc_emb)                               # [1, 1, 768]
            cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)                   # [2, 1, 768]

            vae_emb = embeddings[1].repeat(batch_size, 1, 1, 1)                        # [1, 4, 32, 32]
            vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)                # [2, 4, 32, 32]

        noise_pred = self.unet(
            torch.cat([x_in, vae_emb], dim=1),
            t_in.to(self.unet.dtype),
            encoder_hidden_states=cc_emb,
        ).sample

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        # print(noise_pred_cond.sum().item(), '\t', noise_pred_uncond.sum().item())
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        grad = w * (noise_pred - noise)
        # grad = torch.nan_to_num(grad)

        target = (latents - grad)
        loss = 0.5 * F.l1_loss(latents.float(), target, reduction='sum') / (self.cfg.batch_size * self.cfg.accumulate_grad_batches)

        return loss

    def encode_imgs(self, imgs, mode=False):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample() 
        latents = latents * self.vae.config.scaling_factor

        return latents
    
    def setup_model(self):

        self.pipe.image_encoder.eval()
        self.vae.eval()
        self.unet.train()
        self.clip_camera_projection.eval()
        
        print('[INFO] Training', self.trained_module)

        if self.trained_module == 'unet':
            self.trained_module = self.unet
        elif self.trained_module == 'mid_block':
            self.trained_module = self.unet.mid_block
        elif self.trained_module == 'attentions':
            self.trained_module = self.unet.mid_block.attentions

        self.set_train(self, train=False)
        self.set_train(self.trained_module, train=True)
        self.set_train(self.clip_camera_projection, train=True)
    
    def set_train(self, module, train=True):
        for p in module.parameters():
            p.requires_grad = train
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight)
    
if __name__ == '__main__':
    device = torch.device('cuda')
    
    zero123 = Zero123(device, model_key='ashawkey/zero123-xl-diffusers')