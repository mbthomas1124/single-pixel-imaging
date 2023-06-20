# Single-pixel Image Reconstruction Using Coherent Nonlinear Optics

This repository contains the code and data used in the 2023 paper "Single-pixel Image Reconstruction Using Coherent Nonlinear Optics" by Matthew Thomas, Santosh Kumar, and Yuping Huang. The abstract for this paper is as follows:

> We propose and experimentally demonstrate a novel hybrid optoelectronic system that utilizes mode-selective frequency upconversion, single-pixel detection, and a deep neural network to achieve the reliable reconstruction of 2D images from a noise-contaminated database of handwritten digits. Our system is designed to maximize the multi-scale structural similarity index measure (MS-SSIM) and minimize the mean absolute error (MAE) during the training process. Through extensive evaluation, we have observed that the reconstructed images exhibit high-quality results, with a peak signal-to-noise ratio (PSNR) reaching approximately 20 dB and a structural similarity index measure (SSIM) of around 0.85. These impressive metrics demonstrate the effectiveness and fidelity of our image reconstruction technique. The versatility of our approach allows its application in various fields, including Lidar, compressive imaging, volumetric reconstruction, and so on.

In the "code" folder, one will find the different Python and Jupyter Notebook files used to process data, train models, and produce test results. The "data" folder contains both the simulated and experimentally collected photocurrent features that are used for the image reconstructions, as well as the actual MNIST images that we use as labels. They are organized by crystal position and have already been organized into training and testing set according to a random 80/20 split. The names of the data files give information on their crystal position, the overall noise SNR applied to the input impages (eg. SNR15 denotes an SNR of -15 dB), and the cluster size of this noise (eg. s5 denotes a 5x5 noise clusters).
