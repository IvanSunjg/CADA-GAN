# Context-Aware GAN with Feature Augmentation

### 0 -- Father
### 1 -- Mother
### 2 -- Child (daughter/son)

## Installation
First, ensure you have an environment that functions with python=3.7.4. E.g create one in conda using:
```bash
conda create -n envname python=3.7.4
```

Download the repository and unzip in a location of your choice.
```bash
git clone https://github.com/IvanSunjg/ETH_DL_2022
```

Once finished, please install the following requirements file.
```bash
pip install -r requirement.txt
```
In case torch is not installed properly, try the following command:
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

#### For image segmentation, please enter the following lines in your command prompt, starting from the working directory.
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
Note that the model weights for the pix2pix GAN have to be downloaded separately using the following link: https://polybox.ethz.ch/index.php/s/Kku0HvLrQS1UFC1.
You can also find the resnet/rtnet weights there, in case git lfs was unable to properly download them.

#### For image augmentation, you can enter the following lines in your command prompt.

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

You have to download and extract the folder 'pretrained' in the working directory. The folder can be downloaded from here: https://polybox.ethz.ch/index.php/s/Kku0HvLrQS1UFC1.


## Running

Example: run with augmentation (mixup) and color segmentation
```bash
python main.py --augment --mixup --segment 3 --model 'ImageSegmentation/pix2pixGAN/models/model_seg2_256.h5'
```
