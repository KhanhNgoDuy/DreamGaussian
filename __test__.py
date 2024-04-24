import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import pathlib
from pathlib import Path

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid


model_path = '/mnt/HDD3/khanh/DreamGaussian/loss/models/hair_segmenter.tflite'
# image_paths = list(Path('/mnt/HDD3/khanh/DreamGaussian/images/tony').glob('*.png'))
image_paths = ['/mnt/HDD3/khanh/DreamGaussian/images/tony/0_0.png']

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white

DESIRED_HEIGHT = 256
DESIRED_WIDTH = 256

# Performs resizing and showing the image
def resize(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    return img

class MediapipeProcessor:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a image segmenter instance with the image mode:
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            output_category_mask=True)
        
        self.segmenter = ImageSegmenter.create_from_options(options)
        
    def extract(self, image_path, threshold=0.1):

        if isinstance(image_path, pathlib.PosixPath):
            image_path = image_path.as_posix()

        numpy_image = cv2.imread(image_path)
        numpy_image = resize(numpy_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

        segmentation_result = self.segmenter.segment(mp_image)
        confidence_masks = segmentation_result.confidence_masks
        bg, fg = [mask.numpy_view() for mask in confidence_masks]

        rgb = 1 - mp_image.numpy_view()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        mask = (np.expand_dims(fg, axis=-1) > threshold).astype(np.uint8)
        # output_image = np.where(mask, rgb, np.ones(rgb.shape))
        # output_image = (output_image * 255).astype(np.uint8)
        
        return rgb, mask


processor = MediapipeProcessor()
pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16")
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

prompt = "A human's head with detailed hairstyle"
generator = torch.Generator(device="cuda").manual_seed(0)

for path in image_paths:
    rgb, mask = processor.extract(path)
    rgb = torch.from_numpy(rgb).to(torch.float16).cuda()
    mask = torch.from_numpy(mask).to(torch.float16).cuda()
    image = pipe(
        prompt=prompt, 
        image=rgb, 
        mask_image=mask, 
        strength=0.85,
        num_inference_steps=10,
        guidance_scale=7.5,
    ).images[0]
    print()
