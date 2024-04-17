import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import vflip
from torchvision import transforms

import random
from pathlib import Path
from PIL import Image


class LightningDataWrapper(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root = cfg.root
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers

        if cfg.dataset == "h3ds_v1":
            self.dataset = H3DSv1

    def setup(self, stage):
        full_ds = self.dataset(self.cfg)

        if stage == 'fit':
            train_set_size = int(len(full_ds) * 0.8)
            val_set_size = int(len(full_ds) * 0.1)
            test_set_size = len(full_ds) - train_set_size - val_set_size

            self.train, self.validate, self.test = random_split(full_ds, [train_set_size, val_set_size, test_set_size])
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class H3DSv1(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.root = Path(cfg.root)
        self.batch_size = cfg.batch_size
        self.transform = cfg.transform
        self.dtype = cfg.dtype
        self.device = cfg.device

        self.data = []

        self._prepare_data()
    
    def _prepare_data(self):
        for folder in self.root.iterdir():
            subject = []

            for file in folder.glob('*.png'):
                path = file.as_posix()
                subject.append(path)

            self.data.append(subject)
    
    def get_T(self, target_str, cond_str):
        """
        Convert azimuth, elevation and radius to features.
        Typically, Zero123 uses [d_azim, sin(d_elev), cos(elev), d_radius] as conditional inputs,
        with `d_azim` and `d_elev` being the difference of 2 azims and 2 elevs respectively.
        """
        elev_target, azim_target, r_target = self.get_coords(target_str)
        elev_cond, azim_cond, r_cond = self.get_coords(cond_str)

        d_elev = np.deg2rad(elev_target) - np.deg2rad(elev_cond)
        d_azim = np.deg2rad(azim_target) - np.deg2rad(azim_cond)
        d_r = np.deg2rad(r_target) - np.deg2rad(r_cond)
        
        T = np.stack([d_elev.item(), np.sin(d_azim).item(), np.cos(d_azim.item()), d_r.item()], axis=-1)
        T = torch.from_numpy(T).unsqueeze(1).to(dtype=self.dtype) # [8, 1, 4]       [1, 1, 4]

        return T

    def __getitem__(self, idx):
        subject = self.data[idx]
        out = []

        ## get pair of images and poses
        target_idx, cond_idx = [random.randint(0, len(subject)) for _ in range(2)]
        target_str = subject[target_idx]
        cond_str = subject[cond_idx]

        # load input and conditional images
        for path in [target_str, cond_str]:
            img_tensor = transforms.ToTensor()(Image.open(path)).to(self.dtype)
            img_tensor = vflip(img_tensor)      # Vertically flip for this dataset
            if self.transform:
                img_tensor = self.transform(img_tensor)
            out.append(img_tensor)

        # get pose difference
        T = self.get_T(target_str, cond_str)
        out.append(T)
        
        return out

    def __len__(self):
        return len(self.data)

    def get_coords(self, string):
        """
        Get information of azim, elev, radius from given string
        """
        radius = 2.33   # hard-coded radius to render h3ds dataset
        data = string.split('.')[0].split('/')[-1]
        elev, azim = data.split('_')
        elev, azim = float(elev), float(azim)

        return [elev, azim, radius]


class TestConfig:
    root = "/mnt/HDD2/khanh/content/projects/dreamgaussian/train_zero123/datasets/h3ds_v1"
    batch_size = 16
    num_workers = 2
    dataset = 'h3ds_v1'
    transform = None
    dtype = torch.float16
    device = torch.device('cuda')


if __name__ == '__main__':
    cfg = TestConfig()
    # ds = LightningDataWrapper(cfg)
    # ds.prepare_data()
    # ds.setup('train')

    ds = H3DSv1(cfg)
    loader = DataLoader(ds)

    for data in loader:
        target, cond, T = data
        print(f'{target.shape}\t{cond.shape}]\t{T.shape}')

