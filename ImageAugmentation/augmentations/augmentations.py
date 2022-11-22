import numpy as np
from . import utils


class ItemTransform():

    '''
    An abstract class that is used to indicate when a transform is to be transformed on an item,
    i.e. an image-target pair. Used by transforms including, but not limited to, MixUp and CutMix.
    '''

    def __call__(self, img, label):
        raise NotImplementedError

def apply_transform(t, img, label=None):

    if isinstance(t, ItemTransform):
        img, label = t(img, label)
    else:
        img = t(img)

    return img, label

class _Mix():

    '''
    Superclass for transformations that require image mixing.
    '''

    def __init__(self, dataset):
        self.dataset = dataset

    def get_mix_item(self):
        mix_idx = np.random.randint(len(self.dataset))
        mix_image, mix_target = self.dataset[mix_idx]

        return mix_image, mix_target

class AugMix():
    '''
    @article{hendrycks2020augmix,
    title={{AugMix}: A Simple Data Processing Method to Improve Robustness and Uncertainty},
    author={Hendrycks, Dan and Mu, Norman and Cubuk, Ekin D. and Zoph, Barret and Gilmer, Justin and Lakshminarayanan, Balaji},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2020}
    }

    k: number of different augumentations taken (default 3)
    w1,w2,w3: weight for each augumentated image to mixup
    m: weight for mix with the original and the mixup augumentated image
    level: level of augmentation
    '''

    def __init__(self, k=3, w=[0.2, 0.3, 0.5], m=0.2, level=3):
        self.k = k
        self.w = w
        self.m = m
        self.level = level

    def __call__(self, img):
        '''
        Args:
            img (Tensor): Tensor image of size (C, H, W)
        '''

        # TODO could modify different augmentation method hyperparameters
        miximg = utils.augmix(img, k=self.k, w=self.w, m=self.m, level=self.level)
        return miximg

    def __repr__(self):
        return self.__class__.__name__ + f'(k={self.k}, w={self.w}, m={self.m}, level={self.level})'

class MixUp(_Mix, ItemTransform):

    def __init__(self, dataset, alpha=0.2, min_lam=0.3, max_lam=0.7):
        super().__init__(dataset)

        self.alpha = alpha
        self.min_lam = min_lam
        self.max_lam = max_lam

    def __call__(self, img, label):
        mix_image, mix_label = super().get_mix_item()
        img, label = utils.mixup(img, label, mix_image, mix_label, alpha=self.alpha, min_lam=self.min_lam, max_lam=self.max_lam)

        return img, label

    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, min_lam={self.min_lam}, max_lam={self.max_lam})'