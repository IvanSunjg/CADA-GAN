# Context-Aware GAN with Feature Augmentation

To be continued


## Installation
First, ensure you have an environment that functions with python=3.9.

Download the repository and unzip in a location of your choice.
```bash
git clone https://github.com/IvanSunjg/ETH_DL_2022
```

Once finished, please install the following requirements file.
```bash
pip install -r requirement.txt
```

For image segmentation, please enter the following lines in your command prompt, starting from your working directory.
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
cd ../
cd ../
cd ../
```

For image augmentation, please enter the following lines in your command prompt.
```bash
```

For the GAN pipeline, please enter the following lines in your command prompt.
```bash
```

## Running

Example: run with augmentation (mixup) and color segmentation
```bash
python main.py --augment --mixup --segment 1
```
