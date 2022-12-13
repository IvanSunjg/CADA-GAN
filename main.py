# This is the run script for the final pipeline version.
# Please make sure every dependency is installed in the requirements.txt file.

import argparse
import logging
from augmentations import augmentations as A
from augmentations.TSKinFace_Dataset import TSKinDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


def main(args):

    # Branch when you decide to use the Image Augmentation.
    # MixUp
    if args.augment and args.mixup:
        # logging.info("Image Augmentation is being used. Method is MixUp. Probability is "+str(args.P))
        default_dataset = ImageFolder(
            root = 'dataset/TSKinFace_Data/TSKinFace_cropped/'+args.d+'/',
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        dataset = TSKinDataset(
            root = 'dataset/TSKinFace_Data/TSKinFace_cropped/'+args.d+'/',
            transform = [
                transforms.ToTensor(),
                A.P(A.MixUp(dataset=default_dataset), p=args.P)
            ]
        )
    # AugMix
    elif args.augment and args.augmix:
        # logging.info("Image Augmentation is being used. Method is AugMix. Probability is "+str(args.P))
        dataset = TSKinDataset(
            root = 'dataset/TSKinFace_Data/TSKinFace_cropped/'+args.d+'/',
            transform = [
                transforms.ToTensor(),
                A.P(A.AugMix(), p=args.P)
            ]
        )
    # SmartAugment
    # TODO: Jiugeng Sun
    elif args.augment and args.smart:
        pass


    # Branch when you decide to use the Image Segmentation.
    # TODO: Sofie Daniëls
    else:
        pass

    # GAN Piepline
    # TODO: Jiaqing Xie



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Include the arguments you want to experiment here
    ## TODO: Sofie Daniëls and Jiaqing Xie
    parser.add_argument("-augment", action=argparse.BooleanOptionalAction, help="argument to decide if you are going to use Image Augmentation", type=bool)
    parser.add_argument("-mixup", action=argparse.BooleanOptionalAction, help="argument to decide if you are going to use Mixup", type=bool)
    parser.add_argument("-augmix", action=argparse.BooleanOptionalAction, help="argument to decide if you are going to use AugMix", type=bool)
    parser.add_argument("-p","--P", default=1.0, help="argument to decide the propability of using the Augmentation",type=float)
    parser.add_argument("-smart", action=argparse.BooleanOptionalAction, help="argument to decide if you are going to use SmartAugment", type=bool)

    # Argument to decide which dataset to use
    parser.add_argument("-dataset","--d",default="FMD", help="argument to decide which dataset to use. Default setting is FMD", type=str)

    # Parsing the argument to the args
    args = parser.parse_args()

    # Include some very basic sanity check here to make sure the codes does not crash inside main()
    ## TODO: Sofie Daniëls and Jiaqing Xie
    if args.P > 1.0 or args.P < 0.0:
        raise ValueError("The Probability for argument -p has to be between 0.0 and 1.0!!")

    if not args.augment:
        if args.mixup or args.augmix or args.smart:
            raise ValueError("MixUp, AugMix or SmartAugment can only be used when you set the Augment flag!!!")
    
    if args.augment:
        if not args.mixup and not args.augmix and not args.smart:
            raise ValueError("One of MixUp, AugMix or SmartAugment has to be specified for your augmentation task!!!")
    
    if args.d not in ['FMD', 'FMSD', 'FMS']:
        raise ValueError("Invalid argument setting for dataset. Only FMD, FMS or FMSD accepted!!")

    main(args)
