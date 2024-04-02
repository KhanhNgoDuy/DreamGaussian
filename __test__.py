import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from loss.vgg_face.vgg_face_dag import VGGLoss


vgg_face_loss = VGGLoss('loss/models/vgg_face_dag.pth').to('cuda')
img1 = Image.open('img1.jpg')
img2 = Image.open('img2.jpg')
img3 = Image.open('img3.jpg')
img4 = Image.open('img4.jpg')
imgs = [img1, img2, img3, img4]

transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.PILToTensor(),
]) 

def func_list(input_list, func):
    output_list = []
    for item in input_list:
        output_list.append(func(item))
    return output_list

img1, img2, img3, img4 = [img.unsqueeze(0).float().cuda() for img in func_list(imgs, transform)]

# same id
print(vgg_face_loss(img1, img2))
print(vgg_face_loss(img1, img3))
print(vgg_face_loss(img2, img3))

# different id
print(vgg_face_loss(img4, img1))
print(vgg_face_loss(img4, img2))
print(vgg_face_loss(img4, img3))