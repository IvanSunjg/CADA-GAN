# Context-Aware GAN with Feature Augmentation

To be continued

### 0 -- Father
### 1 -- Mother
### 2 -- Child (daughter/son)


## Installation
First, ensure you have an environment that functions with python=3.9. E.g create one in conda using:
```bash
conda create -n envname python=3.9
```

Download the repository and unzip in a location of your choice.
```bash
git clone https://github.com/IvanSunjg/ETH_DL_2022
```

Once finished, please install the following requirements file.
```bash
pip install -r requirement.txt
```

#### For image segmentation, please enter the following lines in your command prompt, starting from your working directory.
```bash
cd ImageSegmentation
cd face_parsing
git lfs pull
pip install -e .
git clone https://github.com/hhj1897/face_detection
cd face_detection
git lfs pull
pip install -e .
cd ../
git clone https://github.com/hhj1897/roi_tanh_warping
cd roi_tanh_warping
pip install -e .
cd ../../../
```

#### For image augmentation, please enter the following lines in your command prompt.

The library can be installed from GitHub using `pip`.

For Linux and MacOS:
```bash
pip install 'git+https://github.com/IvanSunjg/ETH_DL_2022.git#egg=augmentations&subdirectory=ImageAugmentation'
```

For Windows:
```bash
pip install 'git+https://github.com/IvanSunjg/ETH_DL_2022.git#egg=augmentations^&subdirectory=ImageAugmentation'
```
For more details of the Image Augmentation, please check the Readme file inside the ImageAugmentation folder.

## GAN

### 1. Pretrained VGG:
We take the advantage of pretrained VGG model from https://drive.google.com/drive/folders/1Zo5MV8DMzO_fn3gOMzrUvBxF7oKlN6dN (torch_version). You can download one there or download https://drive.google.com/drive/folders/1aAYxZMKkESr32mNxIrqKx6sh-D8vytdH.  You should place it under `pretrain_stylegan` folder under the root.

### 2. Pretrained High-resolution
We are considering about changing the 


#### For the GAN pipeline, please enter the following lines in your command prompt.
```bash
```



## Running

Example: run with augmentation (mixup) and color segmentation
```bash
python main.py --augment --mixup --segment 1 --model 'ImageSegmentation/pix2pixGAN/models/model_seg2.h5'
```
