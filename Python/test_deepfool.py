import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os

net = models.resnet34(weights='IMAGENET1K_V1')

# Switch to evaluation mode
net.eval()

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'images/test_im5.jpg')

# Verify file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

print(f"Loading image from: {image_path}")
im_orig = Image.open(image_path)

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]


# Remove the mean
im = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])(im_orig)

# do the deepfoolery
r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

labels = open(os.path.join(script_dir, 'synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[int(label_orig)].split(',')[0]
str_label_pert = labels[int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)

tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                        transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224)])

plt.figure()
plt.subplot(1, 2, 1)
# display original image
plt.imshow(im_orig)
plt.title(str_label_orig)
plt.subplot(1, 2, 2)
# display perturbed image
plt.imshow(tf(pert_image.cpu()[0]))
plt.title(str_label_pert)
plt.show()
