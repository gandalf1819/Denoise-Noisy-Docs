# Denoising Noisy Documents

## Autoencoder architecture

![Autoencoder architecture](https://github.com/gandalf1819/Denoise-docs-CV/blob/master/results/Autoencoder.png)

The network is composed of **5 convolutional layers** to extract meaningful features from images. In the first four convolutions, we use **64 kernels**. Each kernel has different weights, perform different convolutions on the input layer, and produce a different feature map. Each output of the convolution, therefore, is composed of 64 channels. 

The encoder uses **max-pooling** for compression. A sliding filter runs over the input image, to construct a smaller image where each pixel is the max of a region represented by the filter in the original image. The decoder uses **up-sampling** to restore the image to its original dimensions, by simply repeating the rows and columns of the layer input before feeding it to a convolutional layer.

**Batch normalization** reduces covariance shift, that is the difference in the distribution of the activations between layers, and allows each layer of the model to learn more independently of other layers.

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

## Data:

Our data is collected from UCI's machine learning repository. The dataset comprises of train and test images of Noisy documents which contain noise from various sources like accidental spills, creases, ink spots and so on. 

[1] https://archive.ics.uci.edu/ml/datasets/NoisyOffice<br>
[2] https://www.kaggle.com/sthabile/noisy-and-rotated-scanned-documents

## Team

* [Kartikeya Shukla](https://github.com/kart2k15)
* [Chinmay Wyawahare](https://github.com/gandalf1819)
* [Michael Lally](https://github.com/MichaelLally)
