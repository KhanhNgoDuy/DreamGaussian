import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image

import torch
import torch.nn.functional as F

import trimesh
import rembg

from cam_utils import orbit_camera, OrbitCamera
from mesh_renderer import Renderer

# from kiui.lpips import LPIPS
from loss.vgg_face.vgg_face_dag import VGGLoss
from guidance.zero123_utils import Zero123

from time import perf_counter
import random


class Config:
    img = "/mnt/HDD3/khanh/DreamGaussian/images/tony/0_270.png"
    ref_size = 256
    H = W = 800
    vers = [0]
    # hors = list(range(30, 91, 30)) # + list(range(5, -95, -5))
    # hors = [90, -90, 180]
    hors = [-5, -15, -20]
    n_samples = 3


class Sampler:

    def __init__(self, cfg):
        self.guidance_zero123 = Zero123('cuda', model_key='ashawkey/stable-zero123-diffusers')

        self.bg_remover = None
        self.load_input(cfg.img)
        self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to('cuda')
        self.input_img_torch = F.interpolate(self.input_img_torch, (cfg.ref_size, cfg.ref_size), mode="bilinear", align_corners=False)
        self.guidance_zero123.get_img_embeds(self.input_img_torch)

        # get samples
        vers = cfg.vers
        hors = cfg.hors
        n_samples = cfg.n_samples
        views = self.sample_zero123(hors, vers, n_samples, save=True)
        # for i, view in enumerate(views):
            # img_avg = torch.stack(view, dim=0).mean(dim=0)
            # img_avg = to_pil_image(img_avg)
            # img_avg.save(f'view {i} avg.jpg')
            # for j, img in enumerate(view):
            #     to_pil_image(img).save(f'view {i}-{j}.jpg')


    def sample_zero123(self, 
                       sample_hors: list = None, 
                       sample_vers: list = None, 
                       n_samples=3, 
                       save=False
                       ):
        hor_min, hor_max, hor_step = (120, 180, 30)
        # ver_min, ver_max, ver_step = (-85, 0, 25)
        self.sample_res = 256    # do not change this value
        num_images_per_prompt = n_samples
        self.sample_radius = 0

        if not (sample_hors or sample_vers):
            sample_hors = [0, 30, 60, 90, 180, 210, 240, 270, 300, 330, 360]
            sample_vers = [0]

        print(f'[INFO] Sampling {num_images_per_prompt} images for {len(sample_hors) * len(sample_vers)} views')

        gt_image = F.interpolate(self.input_img_torch, (self.sample_res, self.sample_res), mode="bilinear", align_corners=False)
        views = []
        
        for i, hor in enumerate(sample_hors):
            for j, ver in enumerate(sample_vers):
                print(f'[{ver}, {hor}]')
                view = self.guidance_zero123.sample(
                    gt_image, [ver], [hor], [self.sample_radius], 
                    self.sample_res, self.sample_res, num_images_per_prompt=num_images_per_prompt  
                )
                views.append(view)

                if save:

                    img_avg = torch.stack(view, dim=0).mean(dim=0)
                    img_avg = to_pil_image(img_avg)
                    img_avg.save(f'sample_{ver}_{hor}_avg.jpg')
                    for k, sample in enumerate(view):
                        to_pil_image(sample.squeeze(0)).save(f'sample_{ver}_{hor}_{k}.jpg')
                    print(f'Saved to sample_{ver}_{hor}.jpg')

        return views

    # Convert a [H, W] tensor to masked PIL Image
    def to_mask(self, tensor):
        mask = tensor.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        image = Image.fromarray(mask)
        return image
    
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        # img = cv2.flip(img, -1)
        # cv2.imwrite('test.jpg', img)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(
            img, (cfg.W, cfg.H), interpolation=cv2.INTER_AREA
        )
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (
            1 - self.input_mask
        )
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()



if __name__ == "__main__":
    cfg = Config()
    Sampler(cfg)