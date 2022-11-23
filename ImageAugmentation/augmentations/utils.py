import torch
from torchvision import transforms
import numpy as np

def mixup(image1, image2, alpha=2, min_lam=0, max_lam=1):
    # Select a random number from the given beta distribution
    # Mixup the images accordingly
    lam = np.clip(np.random.beta(alpha, alpha), min_lam, max_lam) 
    mixup_image = lam * image1 + (1 - lam) * image2

    return mixup_image


def augmix(img, k=3, w=[0.2, 0.3, 0.5], m=0.2, level=2):
    '''
    @article{hendrycks2020augmix,
    title={{AugMix}: A Simple Data Processing Method to Improve Robustness and Uncertainty},
    author={Hendrycks, Dan and Mu, Norman and Cubuk, Ekin D. and Zoph, Barret and Gilmer, Justin and Lakshminarayanan, Balaji},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2020}
    }

    k: number of different augmentations taken (default 3)
    w: list of weights for each augmentation to mixup
    m: weight of the original image when mixing it with the mixed augmentation image
    level: level of augmention
    '''
    if k != len(w):
        raise ValueError(f'k={k} must match the length of w={len(w)}!')

    auglist = ["hflip", "vflip", "autocontrast", "rotate", "translate_x", "translate_y", "shear_x", "shear_y"]
    augs = np.random.choice(auglist, k)
    images = []
    for aug in augs:
        if aug == "hflip":
            new_image = transforms.functional.hflip(img)
        elif aug == "vflip":
            new_image = transforms.functional.vflip(img)
        elif aug == "autocontrast":
            new_image = transforms.functional.autocontrast(img)
        elif aug == "rotate":
            # small rotation degree in order to keep the image from being destroyed
            new_image = transforms.functional.rotate(img, np.random.randint(-10 * level, 10 * level))
        elif aug == "translate_x":
            new_image = transforms.functional.affine(img, translate=(np.random.uniform(-10 * level, 10 * level), 0), angle=0, scale=1, shear=0)
        elif aug == "translate_y":
            new_image = transforms.functional.affine(img, translate=(0, np.random.uniform(-10 * level, 10 * level)), angle=0, scale=1, shear=0)
        elif aug == "shear_x":
            new_image = transforms.functional.affine(img, translate=(0, 0), angle=0, scale=1, shear=(np.random.uniform(-10 * level, 10 * level), 0))
        elif aug == "shear_y":
            new_image = transforms.functional.affine(img, translate=(0, 0), angle=0, scale=1, shear=(0, np.random.uniform(-10 * level, 10 * level)))

        images.append(new_image)

    mixed = torch.zeros_like(img)
    for i in range(k):
        mixed += torch.mul(images[i], w[i])

    miximg = torch.mul(mixed, 1 - m) + torch.mul(img, m)

    return miximg