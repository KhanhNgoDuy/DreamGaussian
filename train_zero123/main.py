import os
os.environ['HF_HOME'] = '/mnt/HDD3/khanh/temp/'

import sys
sys.path.append('./')

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback

from train_utils import Zero123
from train_zero123.datasets import LightningDataWrapper


seed_everything(42)


def main(cfg):
    model = Zero123(cfg, model_key='ashawkey/zero123-xl-diffusers')
    dataset = LightningDataWrapper(cfg)
    # dataset.prepare_data()
    # dataset.setup('fit')

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        devices=1, 
        accelerator=cfg.device,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        deterministic=True,
        callbacks=[PrintCallback()]
    )
    trainer.fit(model, datamodule=dataset)
    trainer.test(model, dataset)


class TestConfig:
    max_epochs = 5 
    trained_module = 'mid_block'
    root = "/mnt/HDD3/khanh/DreamGaussian/train_zero123/datasets/h3ds_v1"
    batch_size = 4
    num_workers = 6
    dataset = 'h3ds_v1'
    transform = None
    dtype = torch.float16
    device = 'gpu'
    accumulate_grad_batches = 3
    num_step_per_val = 2
    learning_rate = 1.0e-04


class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


if __name__ == '__main__':
    cfg = TestConfig()
    main(cfg)