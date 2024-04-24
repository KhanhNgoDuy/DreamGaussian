import os
os.environ['HF_HOME'] = '/mnt/HDD3/khanh/temp/'

import sys
sys.path.append('./')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from diffusers import DDIMScheduler
from einops import rearrange

from zero123 import Zero123Pipeline


class Zero123(nn.Module):
    def __init__(self, cfg, fp16=True, t_range=[0.02, 0.98], model_key="ashawkey/zero123-xl-diffusers", trained_module='attentions'):
        super().__init__()

        self.cfg = cfg
        self._device = 'cuda'
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32
        assert self.fp16, 'Only zero123 fp16 is supported for now.'

        self.trained_module = cfg.trained_module or 'attentions'
        assert self.trained_module in ['attentions', 'mid_block', 'unet']

        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.accumulate_grad_batches = cfg.accumulate_grad_batches

        if model_key == 'ashawkey/stable-zero123-diffusers':
            self.use_stable_zero123 = True
        else:
            self.use_stable_zero123 = False

        self.pipe = Zero123Pipeline.from_pretrained(
            model_key,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self._device)

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.clip_camera_projection = self.pipe.clip_camera_projection
        self.setup_model()

        self.pipe.set_progress_bar_config(disable=True)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        # self.min_step = int(self.num_train_timesteps * t_range[0])
        # self.max_step = int(self.num_train_timesteps * t_range[1])
        self.min_step = 0
        self.max_step = 1
        self.alphas = self.scheduler.alphas_cumprod.to(self._device) # for convenience

        self.step = 0
        self.num_step_per_val = cfg.num_step_per_val

    def get_input(self, data, uncond=0.05):
        target, cond, T = data
        elev, azimuth, radius = T[:, 0].cpu(), T[:, 1].cpu(), T[:, 2].cpu()
        # elev, azimuth, radius = np.array(T.tolist()[0])
        T = self.get_cam_embeddings(elev, azimuth, radius)
        target, cond, T = target.to(self.dtype), cond.to(self.dtype), T.to(self.dtype)
        batch_size, C, H, W = target.shape        # [1, 3, 128, 128]
        
        target = F.interpolate(target, (256, 256), mode='bilinear', align_corners=False)
        cond = F.interpolate(cond, (256, 256), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(target)
        
        # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        random = torch.rand(batch_size, device=target.device)
        # input_mask = 1 - rearrange((random >= uncond).half() * (random < 3 * uncond).half(), "n -> n 1 1 1")

        feat = {}
        clip_emb = self.get_learned_conditioning(cond).detach()
        feat["c_crossattn"] = self.clip_camera_projection(torch.cat([clip_emb[:, None], T], dim=-1))
        # feat["c_concat"] = input_mask * self.encode_imgs(cond) / self.vae.config.scaling_factor
        feat["c_concat"] = self.encode_imgs(cond) / self.vae.config.scaling_factor
        
        return [latents, feat]
            
    def forward(self, data, step_ratio=None, guidance_scale=5):
        # rgb: tensor [1, 3, H, W] in [0, 1]
        rgb, rgb_cond, T = data
        batch_size, C, H, W = rgb.shape        # [1, 3, 128, 128]

        assert step_ratio == None    
        latents, feat = self.get_input(data)

        # Add noise to target image

        self.step += 1
        step_ratio = min(1, self.step / self.cfg.max_epochs)
        t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
        t = torch.full((batch_size,), t, dtype=torch.long, device=self._device)
        # t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self._device)

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)                  # [N, 1, 1, 1]
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)     # False

        c_concat = feat["c_concat"]                                     # False
        c_crossattn = feat["c_crossattn"]                               # True
        x_in = torch.cat([latents_noisy, c_concat], dim=1)              # False
        cc_emb = c_crossattn

        # gradient for unet
        noise_pred = self.unet(
            x_in,
            t.to(self.unet.dtype),
            encoder_hidden_states=cc_emb,
        ).sample

        grad = w * (noise_pred - noise)
        # grad = torch.nan_to_num(grad)

        target = (latents - grad)
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / (self.batch_size * self.accumulate_grad_batches)
        # print(loss.item())

        return loss
    
    def get_cam_embeddings(self, elevation, azimuth, radius, default_elevation=0):
        if self.use_stable_zero123:
            T = np.stack([np.deg2rad(elevation), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), np.deg2rad([90 + default_elevation] * len(elevation))], axis=-1)
        else:
            # original zero123 camera embedding
            T = np.stack([np.deg2rad(elevation), np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth)), radius], axis=-1)
        T = torch.from_numpy(T).unsqueeze(1).to(dtype=self.dtype, device=self._device) # [8, 1, 4]       [1, 1, 4]
        return T

    def get_learned_conditioning(self, cond):
        x_pil = [TF.to_pil_image(image) for image in cond]
        x_clip = self.pipe.feature_extractor(images=x_pil, return_tensors="pt").pixel_values.to(device=self._device, dtype=self.dtype)
        c = self.pipe.image_encoder(x_clip).image_embeds        # [1, 768]
        return c

    def encode_imgs(self, imgs, mode=False):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() 
        latents = latents * self.vae.config.scaling_factor

        return latents
    
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs
        
    # def training_step(self, batch, batch_idx):
    #     loss = self(batch)
    #     return loss
    
    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx):
    #     loss = self(batch)
    #     return loss
        
    # @torch.no_grad()
    # def test_step(self, batch, batch_idx):
    #     loss = self(batch)
    #     if self.step % self.num_step_per_val == 0:
    #         rgb, rgb_cond, T = batch
    #         elev, sin_azim, radius = T
    #         azim = np.arcsin(sin_azim)
    #         elev, azim = np.rad2deg(elev), np.rad2deg(azim)

    #         img_list = self.sample(rgb, elev, azim, radius, height=256, width=256, num_images_per_prompt=1)
    #         for img in img_list:
    #             save_name = f"step{self.step}_elev{elev}_azim{azim}.jpg"
    #             TF.to_pil_image(img).save(save_name)
    #     return loss

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

    def sample(self, image, elevation, azimuth, distance, height, width, 
               num_inference_step = 50, 
               guidance_scale = 7.5, 
               num_images_per_prompt = 3,
               clip_image_embeddings = None, 
               image_camera_embeddings = None,
               return_dict = True
    ):  
        """
        Return a list of PIL images
        The number of returned images is `num_images_per_prompt`
        """
        elevation = torch.tensor(elevation)
        azimuth = torch.tensor(azimuth)
        distance = torch.tensor(distance)

        images = self.pipe(
            image, elevation, azimuth, distance, height, width, 
            num_inference_step, guidance_scale, num_images_per_prompt, 
            clip_image_embeddings=clip_image_embeddings, 
            image_camera_embeddings=image_camera_embeddings,
            return_dict=return_dict
        )['images']

        img_list = [transforms.ToTensor()(img).float().to(self._device) for img in images]

        return img_list
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.trained_module.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight)


if __name__ == '__main__':
    
    class TestConfig:
        max_epochs = 5 
        trained_module = 'mid_block'
        root = "/mnt/HDD3/khanh/DreamGaussian/train_zero123/datasets/h3ds_v1"
        batch_size = 16
        num_workers = 2
        dataset = 'h3ds_v1'
        transform = None
        dtype = torch.float32
        device = 'gpu'
        accumulate_grad_batches = 16
        num_step_per_val = 2
        learning_rate = 1.0e-04

    cfg = TestConfig()
    model = Zero123(cfg)

    for n, p in model.named_parameters():
        print(f'{n} \t\t {p.shape}')