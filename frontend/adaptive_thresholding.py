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

def adaptive_thresholding(noisy_img):
    adaptive_th=cv2.adaptiveThreshold(noisy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,30)
    return adaptive_th
    
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

# Try an example
denoised=cleaned_img[4]
noisy=dirty_img[4]

plt.figure(figsize=(10,6))
plt.imshow(denoised, cmap='gray')
plt.grid(b=None)

plt.figure(figsize=(10,6))
plt.imshow(noisy, cmap='gray')
plt.grid(b=None)

adaptive_th=adaptive_thresholding(noisy)

plt.figure(figsize=(10,6))
plt.imshow(adaptive_th, cmap='gray')
plt.grid(b=None)

# Compute RMSE
print("RMSE: ", rmse(adaptive_th, denoised))

# Compute UQI
print("UQI:", uqi(sample, predicted_label))