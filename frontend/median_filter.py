import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
from skimage import io
import seaborn as sns
import math
import cv2
from sewar.full_ref import uqi, psnr, rmse, ssim
sns.set()
from scipy import signal
# %matplotlib inline

def median_subtract(noisy_img, ksize=23):
    background=cv2.medianBlur(noisy_img, ksize)
    result=cv2.subtract(background, noisy_img)
    result=cv2.bitwise_not(result)
    return (result, background)
    
# Use boto3 for aws config
# client = boto3.client('s3')
# input_dir = 's3://denoise-docs/'

input_dir = '/Users/chinmay/Documents/GitHub/Denoise-docs-CV/dataset/'

# Load dataset from S3
img_paths=os.listdir(input_dir + 'train_cleaned')
img_paths=['train_cleaned/'+x for x in img_paths]
img_paths.index('train_cleaned/72.png')
cleaned_img=[cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in img_paths]

img_paths=os.listdir(input_dir + 'train')
img_paths=['train/'+x for x in img_paths]
dirty_img=[cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in img_paths]

# Compute median filtering
denoised=cleaned_img[-1]
result, background=median_subtract(dirty_img[-1])

rmsse = 1 

# Compute RMSE
print("RMSE: ", rmse(denoised, result))

# Compute UQI
print("UQI: ", uqi(denoised,result))

# Compute PSNR
print("PSNR: ", psnr(denoised, result))

# Compute SSIM
print("SSIM:", ssim(denoised, result))

# Observe the output
plt.rcParams["axes.grid"] = False
fig, axarr= plt.subplots(2,2, figsize=(15,15))
axarr[0,0].imshow(denoised, cmap='gray')
axarr[0,1].imshow(dirty_img[-1], cmap='gray')
axarr[1,0].imshow(background, cmap='gray')
axarr[1,1].imshow(result, cmap='gray')

# Visualize the cleaned image
font = {'family': 'serif',
    'color':  'darkred',
    'weight': 'normal',
    'size': 16,
}
plt.rcParams["axes.grid"] = False
fig=plt.figure(figsize=(12,8))
plt.imshow(denoised, cmap='gray')
plt.title('Clean Image', fontdict=font)

# Visualize the dirty image
font = {'family': 'serif',
    'color':  'darkred',
    'weight': 'normal',
    'size': 16,
}
plt.rcParams["axes.grid"] = False
fig=plt.figure(figsize=(12,8))
plt.imshow(dirty_img[-1], cmap='gray')
plt.title('Dirty Image', fontdict=font)

# Calculate Background using Median Filter
font = {'family': 'serif',
    'color':  'darkred',
    'weight': 'normal',
    'size': 16,
}
plt.rcParams["axes.grid"] = False
fig=plt.figure(figsize=(12,8))
plt.imshow(background, cmap='gray')
plt.title('Calculated Background using Median Filter', fontdict=font)

# Visualize the result
font = {'family': 'serif',
    'color':  'darkred',
    'weight': 'normal',
    'size': 16,
}
plt.rcParams["axes.grid"] = False
fig=plt.figure(figsize=(12,8))
plt.imshow(result, cmap='gray')
plt.title('Result (after subtracting background from OG image)', fontdict=font)