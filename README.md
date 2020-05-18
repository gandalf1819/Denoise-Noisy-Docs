# Denoising Noisy Documents

![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)
![Python](https://upload.wikimedia.org/wikipedia/commons/3/34/Blue_Python_3.6_Shield_Badge.svg)
[![Build Status](https://travis-ci.org/usgs/nwisweb-tableau-data-connector.svg?branch=master)](https://travis-ci.org/usgs/nwisweb-tableau-data-connector)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)


Numerous scientific papers, historical documentaries/artifacts, recipes, books are stored as papers be it handwritten/typewritten. With time, the paper/notes tend to accumulate noise/dirt through fingerprints, weakening of paper fibers, dirt, coffee/tea stains, abrasions, wrinkling, etc. There are several surface cleaning methods used for both preserving and cleaning, but they have certain limits, the major one being: that the original document might get altered during the process. The purpose of this project is to do a comparative study of traditional computer vision techniques vs deep learning networks when denoising dirty documents.


## Autoencoder architecture

![Autoencoder architecture](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/Autoencoder-pipeline.png)

The network is composed of **5 convolutional layers** to extract meaningful features from images. In the first four convolutions, we use **64 kernels**. Each kernel has different weights, perform different convolutions on the input layer, and produce a different feature map. Each output of the convolution, therefore, is composed of 64 channels. 

The encoder uses **max-pooling** for compression. A sliding filter runs over the input image, to construct a smaller image where each pixel is the max of a region represented by the filter in the original image. The decoder uses **up-sampling** to restore the image to its original dimensions, by simply repeating the rows and columns of the layer input before feeding it to a convolutional layer.

**Batch normalization** reduces covariance shift, that is the difference in the distribution of the activations between layers, and allows each layer of the model to learn more independently of other layers.

```
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
image_input (InputLayer)     (None, 420, 540, 1)       0         
_________________________________________________________________
Conv1 (Conv2D)               (None, 420, 540, 32)      320       
_________________________________________________________________
pool1 (MaxPooling2D)         (None, 210, 270, 32)      0         
_________________________________________________________________
Conv2 (Conv2D)               (None, 210, 270, 64)      18496     
_________________________________________________________________
pool2 (MaxPooling2D)         (None, 105, 135, 64)      0         
_________________________________________________________________
Conv3 (Conv2D)               (None, 105, 135, 64)      36928     
_________________________________________________________________
upsample1 (UpSampling2D)     (None, 210, 270, 64)      0         
_________________________________________________________________
Conv4 (Conv2D)               (None, 210, 270, 32)      18464     
_________________________________________________________________
upsample2 (UpSampling2D)     (None, 420, 540, 32)      0         
_________________________________________________________________
Conv5 (Conv2D)               (None, 420, 540, 1)       289       
=================================================================
Total params: 74,497
Trainable params: 74,497
Non-trainable params: 0
_________________________________________________________________
```

## Regression

Along with autoencoder, another machine learning technique we have used is **Linear Regression**. Instead of modelling the entire image at once, we tried predicting the cleaned-up intensity for each pixel within the image, and constructed a cleaned image by combining together a set of predicted pixel intensities using linear regression. Except at the extremes, there is a linear relationship between the brightness of the dirty images and the cleaned images. There is a broad spread of x values as y approaches 1, and these pixels probably represent stains that need to be removed.

![Regression-result](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/regression-results/reg-1.png)

## AWS architecture

![AWS Architecture](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/CV-architecture.png)

## Analysis Approach

1. Use **median filter** to get a “background” of the image, with the text being “foreground” (due to the fact that the noise takes more space than the text in large localities). Next subtract this “background” from the original image.<br>
2. Apply **canny edge detection** to extract edges. Perform dilation (i.e. make text/lines thicker, and noise/lines thinner) then erosion while preserving thicker lines and removing thinner ones (i.e. noisy edges)
3. Use **adaptive thresholding.** (works really well since often text is darker than noise). Thus, preserve pixels that are darkest “locally” and threshold rest to 0 (i.e. foreground)
4. **CNN Autoencoder:** The network is composed of 5 convolutional layers to extract meaningful features from images.
  * During convolutions, same padding mode will be used. We pad with zeros around the input matrix, to preserve the same image dimensions after convolution.
  * The encoder uses max-pooling for compression. A sliding filter runs over the input image, to construct a smaller image where each pixel is the max of a region represented by the filter in the original image.
  * The decoder uses up-sampling to restore the image to its original dimensions, by simply repeating the rows and columns of the layer input before feeding it to a convolutional layer.
  * Perform batch-normalization as required. For the output, we use sigmoid activation to predict pixel intensities between 0 and 1.
5. Compare results from {1, 2, 3, 4} using the following metrics: **RMSE, PSNR, SSIM, UQI**

## Results:

### Median Filtering
|||||
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![Clean](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/median-results/clean.png)  |  ![Dirty](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/median-results/dirty.png)  |  ![Background](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/median-results/background.png)  |  ![Final](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/median-results/final-result.png)

### Adaptive Thresholding

||||
:-------------------------:|:-------------------------:|:-------------------------:
![Results](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/adaptive-results/after-ad-th.png)  |  ![Clean](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/adaptive-results/clean.png)  |  ![Dirty](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/adaptive-results/dirty.png)

### Canny Edge Detection
||||||
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![After](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/edge-detection-results/after-edge-detection.png)  |  ![Clean](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/edge-detection-results/clean.png)  |  ![Dilation](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/edge-detection-results/dilation.png)  |  ![Dirty](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/edge-detection-results/dirty.png)  |  ![Erosion](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/edge-detection-results/final-result-erosion.png)

### Autoencoder

|||
:-------------------------:|:-------------------------:
![Noisy](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/autoencoder-results/noisy.png)  |  ![Trained-Clean](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/autoencoder-results/trained-cleaned.png)

### Linear Regression
|||
:-------------------------:|:-------------------------:
![Noisy](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/regression-results/reg-dirty.png)  |  ![Trained-Clean](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/regression-results/reg-noisy.png)

## Screens

#### Login Screen

|||
:-------------------------:|:-------------------------:
![Login](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/screens/login.png)  |  ![Autoencoder](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/screens/autoencoder-input.png)
Login Screen  |  Autoencoder
![Median](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/screens/median-login.png)  |  ![Median-Results](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/screens/median-results.png)
Median Filtering  |  Median Filtering Results

## Project Schema

Directory structure for Denoizer repository:

```
|-- dataset
 |-- test.zip → test image dataset
	|-- train.zip → training image dataset containing noisy dataset
	|-- train_cleaned.zip → cleaned images for respective noisy images in train.zip
|-- frontend
	|-- static → CSS, JS for flask web app 
	|-- templates → HTML pages for flask web app
|-- reports → collection of reports submitted on this project
|-- results → resultant images for each technique 
	|-- adaptive-results
	|-- autoencoder-results
	|-- edge-detection-results
	|-- median-results
	|-- regression-results
|-- screens → screenshot/snippets  for each tab/page on webapp
```


## Data:

Our data is collected from UCI's machine learning repository. The dataset comprises of train and test images of Noisy documents which contain noise from various sources like accidental spills, creases, ink spots and so on. 

[1] https://archive.ics.uci.edu/ml/datasets/NoisyOffice<br>
[2] https://www.kaggle.com/sthabile/noisy-and-rotated-scanned-documents

## Team

* [Kartikeya Shukla](https://github.com/kart2k15)
* [Chinmay Wyawahare](https://github.com/gandalf1819)
* [Michael Lally](https://github.com/MichaelLally)
