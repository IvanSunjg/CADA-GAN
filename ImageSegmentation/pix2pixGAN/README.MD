# Pix2PixGAN

A code base that undoes image segmentation with the help of Pix2PixGAN. The code is adapted from https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/

## Dependencies

* numpy >= 1.17.0
* cudatoolkit (for GPU use)
* cudnn (for GPU use)
* tensorflow
* matplotlib
* imageio

## Installation

```bash
git clone https://github.com/IvanSunjg/ETH_DL_2022.git
cd ImageSegmentation/pix2pixGAN
```

## Getting Started

Test
```bash
python pix2pixSEG.py -i /path/to/train/folder -e /path/to/test/folder -o /path/to/output/folder
```

Command line arguments
```bash
--train, -i, Path to training images
--test, -e, Path to test images
--output, -o, Path to output folder
--testbatch, -t, Number of images you want to test, has to be smaller than or equal to test dataset size (default=1)
--epochs, -n, Number of epochs you want to train for (default=5),
--batches, -b, Number of batches for training (default=64)
```
