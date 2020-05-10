import json
import boto3
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
from skimage import io
import seaborn as sns
import math
import cv2
from sewar.full_ref import uqi, psnr, rmse, ssim
import pandas as pd
sns.set()
from scipy import signal
%matplotlib inline

def edge_detection_dilation_erosion(noisy_img):
    edges=cv2.Canny(noisy_img, 100, 100)
    edges=cv2.bitwise_not(edges)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(cv2.bitwise_not(edges),kernel,iterations = 1)
    dilation=cv2.bitwise_not(dilation)
    kernel1 = np.ones((5,5),np.uint8)
    erosion=cv2.erode(cv2.bitwise_not(dilation), kernel1,iterations=1)
    erosion=cv2.bitwise_not(erosion)
    return (edges, dilation, erosion)
    
# Use boto3 for aws config
client = boto3.client('s3')
input_dir = 's3://denoise-docs/'

# Load dataset from S3
img_paths=os.listdir(input_dir + 'train_cleaned')
img_paths=['train_cleaned/'+x for x in img_paths]
cleaned_img=[cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in img_paths]

img_paths=os.listdir(input_dir + 'train')
img_paths=['train/'+x for x in img_paths]
dirty_img=[cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in img_paths]

# Try for a sample image
denoised=cleaned_img[0]
noisy=dirty_img[0]

# Compute edges, dilation and erosion
edges, dilation, erosion = edge_detection_dilation_erosion(noisy)

# Observe the output
fig=plt.figure(figsize=(12,8))
plt.imshow(edges, cmap='gray')
plt.grid(b=None)

fig=plt.figure(figsize=(12,8))
plt.imshow(dilation, cmap='gray')
plt.grid(b=None)

fig=plt.figure(figsize=(12,8))
plt.imshow(erosion, cmap='gray')
plt.grid(b=None)

# Compute RMSE
print("RMSE: ", rmse(denoised, erosion))

# Compute UQI
print("UQI: ", uqi(denoised, erosion))