import os

import numpy as np
os.environ['HF_HOME'] = '/mnt/HDD3/khanh/temp/'

import sys
sys.path.append('./')

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from train_utils import Zero123
# from utils import Zero123
from train_zero123.datasets import H3DSv1


def main(cfg):
    model = Zero123(cfg, model_key='ashawkey/zero123-xl-diffusers')
    # for m in model.children():
    #     m.apply(model.init_weights)
    dataset = H3DSv1(cfg)
    optimizer = model.configure_optimizers()
    # optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True    
    )
    
    for epoch in range(cfg.max_epochs):
        _loss = 0
        step = 0

        for idx, (target, cond, T) in enumerate(train_dataloader, start=1):

            step += 1
            step_ratio = min(1, step / cfg.max_epochs)

            target, cond, T = target.to('cuda'), cond.to('cuda'), T.to('cuda')
            # cond = target
            target = F.interpolate(target, (256, 256), mode='bilinear', align_corners=False)
            cond = F.interpolate(cond, (256, 256), mode='bilinear', align_corners=False)
            loss = model([target, cond, T])
            # loss = model(pred_rgb=target, cond_rgb=cond, data=T, step_ratio=step_ratio)
            loss.backward()
            _loss = _loss + loss.item()
            
            # if (idx % cfg.num_step_per_val) == 0 or (idx == 1):
            #     elev_rad, sin_azim_rad, cos_azim_rad, rs = T.squeeze(1).permute(1, 0).tolist()
            #     elevs = np.rad2deg(elev_rad)
            #     azims = np.rad2deg(np.arcsin(sin_azim_rad))
            #     azim_s = np.rad2deg(np.arccos(cos_azim_rad))

            #     _, feat = model.get_input([target, cond, T])
            #     image_camera_embeddings = feat['c_crossattn']

            #     for elev, azim, r in zip(elevs, azims, rs):
            #         imgs = model.sample(
            #             image=target,
            #             elevation=elev,
            #             azimuth=azim,
            #             distance=r,
            #             height=256,
            #             width=256,
            #             image_camera_embeddings=image_camera_embeddings
            #         )
            #         for i, img in enumerate(imgs):
            #             to_pil_image(img).save(f'train_zero123/samples/ep{epoch}_{idx}_{i}.jpg')
            #             print(f'--> train_zero123/samples/ep{epoch}_{idx}_{i}.jpg')

            #     # for (elev, azim, r) in zip(elevs, azims, rs):
            #     for j in range(len(elevs)):
            #         imgs = model.sample(
            #             image=target[j].unsqueeze(0),
            #             elevation=[elevs[j]],
            #             azimuth=[azims[j]],
            #             distance=[rs[j]],
            #             height=256,
            #             width=256,
            #             # image_camera_embeddings=image_camera_embeddings
            #         )
            #         for i, img in enumerate(imgs):
            #             to_pil_image(img).save(f'train_zero123/samples/_ep{epoch}_{idx}_{i}.jpg')
            #             print(f'--> train_zero123/samples/_ep{epoch}_{idx}_{i}.jpg')

            if (idx % cfg.accumulate_grad_batches == 0) or (idx == len(train_dataloader)):
                print(f'Step {idx}, loss: {_loss}')
                grad = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad = grad + p.grad.mean().item()
                print('Before clip:', grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0)
                grad = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad = grad + p.grad.mean().item()
                print('After clip:', grad)
                optimizer.step()
                optimizer.zero_grad()

        print(f'Epoch {epoch} train loss {_loss}')
        break


class TestConfig:
    max_epochs = 10000
    trained_module = 'unet'
    root = "/mnt/HDD3/khanh/DreamGaussian/train_zero123/datasets/h3ds_v1"
    batch_size = 1
    num_workers = 0
    dataset = 'h3ds_v1'
    transform = None
    dtype = torch.float16
    device = 'cuda'
    accumulate_grad_batches = 1
    num_step_per_val = 192
    learning_rate = 1.0e-4


if __name__ == '__main__':
    cfg = TestConfig()
    main(cfg)