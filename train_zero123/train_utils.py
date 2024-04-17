import os
os.environ['HF_HOME'] = '/mnt/HDD2/khanh/temp/'

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


class Zero123(pl.LightningModule):
    def __init__(self, cfg, fp16=True, t_range=[0.02, 0.98], model_key="ashawkey/zero123-xl-diffusers", trained_module='attentions'):
        super().__init__()

        # self.device = cfg.device
        self.fp16 = fp16
        self.datatype = torch.float16 if fp16 else torch.float32
        assert self.fp16, 'Only zero123 fp16 is supported for now.'

        self.trained_module = cfg.trained_module or 'attentions'
        assert self.trained_module in ['attentions', 'mid_block', 'unet']

        self.learning_rate = cfg.learning_rate

        self.pipe = Zero123Pipeline.from_pretrained(
            model_key,
            torch_dtype=self.datatype,
            trust_remote_code=True,
        ).to(self.device)

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.cc_projection = self.pipe.clip_camera_projection
        self.setup_model()

        self.pipe.set_progress_bar_config(disable=True)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        self.step = 0
        self.num_step_per_val = cfg.num_step_per_val

    # @torch.no_grad()
    def get_input(self, data, uncond=0.05):
        target, cond, T = data
        batch_size, C, H, W = target.shape        # [1, 3, 128, 128]
        
        target = F.interpolate(target, (256, 256), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(target.to(self.datatype))
        
        # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        random = torch.rand(batch_size, device=target.device)
        prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
        input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")

        feat = {}
        clip_emb = self.get_learned_conditioning(cond).detach()
        null_prompt = self.get_learned_conditioning([""]).detach()
        feat["c_crossattn"] = [self.clip_camera_projection(torch.cat([torch.where(prompt_mask, null_prompt, clip_emb), T[:, None, :]], dim=-1))]
        feat["c_concat"] = [input_mask * self.encode_imgs(cond) / self.vae.config.scaling_factor]
        
        return [latents, feat]
            
    def forward(self, data, step_ratio=None, guidance_scale=5):
        # rgb: tensor [1, 3, H, W] in [0, 1]
        rgb, rgb_cond, T = data
        batch_size, C, H, W = rgb.shape        # [1, 3, 128, 128]

        assert step_ratio == None    
        latents, feat = self.get_input(data)

        # Add noise to target image
        t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        t_in = torch.cat([t] * 2)

        c_concat = feat["c_concat"]
        c_crossattn = feat["c_crossattn"]
        x_in = torch.cat([latents_noisy, c_concat], dim=1)
        cc_emb = torch.cat(c_crossattn, 1)

        # gradient for unet
        noise_pred = self.unet(
            x_in,
            t_in.to(self.unet.dtype),
            encoder_hidden_states=cc_emb,
        ).sample

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        loss = w * F.mse_loss(noise_pred, noise, reduction='sum')

        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self(batch)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        return loss
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss = self(batch)
        if self.step % self.num_step_per_val == 0:
            rgb, rgb_cond, T = batch
            elev, sin_azim, radius = T
            azim = np.arcsin(sin_azim)
            elev, azim = np.rad2deg(elev), np.rad2deg(azim)

            img_list = self.sample(rgb, elev, azim, radius, height=256, width=256, num_images_per_prompt=1)
            for img in img_list:
                save_name = f"step{self.step}_elev{elev}_azim{azim}.jpg"
                TF.to_pil_image(img).save(save_name)
        return loss

    def setup_model(self):

        self.pipe.image_encoder.train()
        self.vae.train()
        self.unet.train()
        self.cc_projection.train()
        
        if self.trained_module == 'unet':
            self.trained_module = self.unet
        elif self.trained_module == 'mid_block':
            self.trained_module = self.unet.mid_block
        elif self.trained_module == 'attentions':
            self.trained_module = self.unet.mid_block.attentions

        self.set_train(self, train=False)
        self.set_train(self.trained_module, train=True)
        self.set_train(self.cc_projection, train=True)

    def get_learned_conditioning(self, c):
        return self.pipe.image_encoder(c)
    
    def set_train(self, module, train=True):
        for p in module.parameters():
            p.requires_grad = train

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

        img_list = [transforms.ToTensor()(img).float().to(self.device) for img in images]

        return img_list
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.trained_module.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

if __name__ == '__main__':
    model = Zero123(device='cuda')