# This is the run script for the final pipeline version.
# Please make sure every dependency is installed in the requirements.txt file.

import argparse
import logging
import numpy as np
import torch
from augmentations import augmentations as A
from augmentations.TSKinFace_Dataset import TSKinDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from ImageSegmentation.face_parsing.face_parsing_test import face_parsing_test
from ImageSegmentation.pix2pixGAN.test import test_pix2pix
from MLP import FourLayerNet

import matplotlib.pyplot as plt

from PipelineGAN.stylegan_train import load, embedding_function, style_transfer
from PipelineGAN.utils import PSNR, loss_function
from PipelineGAN.VGG16 import VGG16_perceptual


def main(args):

    # Branch when you decide to use the Image Augmentation.
    # MixUp
    if args.augment and args.mixup:
        # logging.info("Image Augmentation is being used. Method is MixUp. Probability is "+str(args.P))
        default_dataset = ImageFolder(
            root = args.data_path + args.dataset+'/',
            transform = transforms.Compose([
                transforms.CenterCrop((1024, 1024)),
                transforms.ToTensor(),
            ])
        )
        if args.dataset == 'FMD':
            dataset = TSKinDataset(
                root = args.data_path + args.dataset+'/',
                transform = [
                    transforms.ToTensor(),
                    transforms.CenterCrop((1024, 1024)),
                    A.P(A.MixUp_FMD(dataset=default_dataset), p=args.P)
                ]
            )
        elif args.dataset == 'FMS':
            dataset = TSKinDataset(
                root = args.data_path + args.dataset+'/',
                transform = [
                    transforms.ToTensor(),
                    transforms.CenterCrop((1024, 1024)),
                    A.P(A.MixUp_FMS(dataset=default_dataset), p=args.P)
                ]
            )
        elif args.dataset == 'FMSD':
            dataset = TSKinDataset(
                root = args.data_path + args.dataset+'/',
                transform = [
                    transforms.ToTensor(),
                    transforms.CenterCrop((1024, 1024)),
                    A.P(A.MixUp_FMSD(dataset=default_dataset), p=args.P)
                ]
            )
 
    # AugMix
    elif args.augment and args.augmix:
        # logging.info("Image Augmentation is being used. Method is AugMix. Probability is "+str(args.P))
        dataset = TSKinDataset(
            root = args.data_path + args.dataset+'/',
            transform = [
                transforms.ToTensor(),
                transforms.CenterCrop((1024, 1024)),
                A.P(A.AugMix(), p=args.P)
            ]
        )
    # SmartAugment
    # TODO: Jiugeng Sun
    elif args.augment and args.smart:
        pass

    # Basic Unaugmented Dataset 
    else:
        dataset = ImageFolder(
            root = args.data_path + args.dataset+'/',
            transform = transforms.Compose([
                transforms.CenterCrop((1024, 1024)),
                transforms.ToTensor(),
            ])
        )
    
    
   # Use Image Segmentation.
    if args.segment:
        data_in = []
        label_list = []
        for images, labels in dataset:
            data_in.append(np.transpose(images.numpy(), (1, 2, 0)))
            label_list.append(labels)
        data_in = np.asarray(data_in)

        # Segment all images (child, mother, father)
        data_out = face_parsing_test(input_images=data_in, blurring=args.segment-1)
        
        # List of original images, segmented images, labels
        data_list_seg = list(zip(data_in, data_out, label_list))      
      
    # GAN projection into latent space
    # TODO: Jiaqing Xie
    if args.gan == "image2stylegan":

        _, g_synthesis = load(args)


        idx = 0
        c_idx = []
        f_idx = []
        m_idx = []

        for image, label in dataset:
            if label == 0:
                f_idx.append(idx)
            elif label == 1:
                m_idx.append(idx)
            elif label == 2:
                c_idx.append(idx)
            idx+=1

        for f, m in zip(f_idx,m_idx):

            # father image
            image_f = dataset[f][0]
            image_f = image_f[None, :]
            image_f = image_f.to(args.device)

            # mother image
            image_m = dataset[m][0]
            image_m = image_m[None, :]
            image_m = image_m.to(args.device)
            
            latent_f = embedding_function(image_f, args, g_synthesis).to(args.device)
            latent_m = embedding_function(image_m, args, g_synthesis).to(args.device)

            break


    # Feature selection
    # TODO: Jiaqing Xie please check if it works
    model_f = FourLayerNet().to(args.device)
    model_m = FourLayerNet().to(args.device)
    loss_f = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(list(model_f.parameters()) + list(model_m.parameters()),  lr=args.lr)




    true_child_img = dataset[c_idx[0]][0]
    true_child_img = true_child_img[None, :]
    true_child_img = true_child_img.to(args.device)
    upsample = torch.nn.Upsample(scale_factor=256 / 1024, mode='bilinear')
    img_p = image.clone()
    img_p = upsample(img_p)


    for i in range(args.epochs):

 
        optim.zero_grad()
        new_latent_f = model_f(latent_f)
        new_latent_m = model_m(latent_m)
        latent = torch.cat((new_latent_f, new_latent_m), dim=2)
        syn_child_img = g_synthesis(latent)

        perceptual = VGG16_perceptual().to(args.device)
        mse, per_loss = loss_function(syn_child_img ,true_child_img, img_p, loss_f, upsample, perceptual)
        psnr = PSNR(mse, flag=0)
        loss = per_loss + mse
        loss.backward()
        optim.step()
        if (i + 1) % 500 == 0:
            print("iter{}: , mse_loss --{}, psnr --{}".format(i + 1, loss, psnr))
            save_image(syn_child_img.clamp(0, 1), "save_images/reconstruct_{}.png".format(i + 1))


    # GAN from latent space back to image
    # TODO: Jiaqing Xie
    syn_child_img = g_synthesis(latent)
    
    # Undo image segmentation
    # TODO: Sofie Daniëls
    # parents → list of tuples: original image, segmented image
    data_list_fathers = []
    data_list_mothers = []
    # children → list of quadruplets: original image, segmented image, generated segmented image
    data_list_GAN_out = []
    if args.segment:
        data_list_seg_gen = []
        data_list_real = []
        data_list_seg = []
        for (orig, seg, gen) in data_list_GAN_out:
            data_list_seg_gen.append(np.transpose(gen.numpy(), (1, 2, 0)))
            data_list_real.append(orig)
            data_list_seg.append(seg)

        data_list_real_gen = test_pix2pix([data_list_seg_gen, data_list_real], args.model, 'Output/')

        # list of octoplets of following images:
        # F orig, F seg, M orig, M seg, C orig, C seg, C generated segmentation, C generated real
        # (With F = father, M = mother, C = child, seg = segmented, orig = original)
        data_list_final = list(zip(data_list_fathers, data_list_mothers, data_list_real, data_list_seg, data_list_seg_gen, data_list_real_gen))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Include the arguments you want to experiment here
    ## TODO: Jiaqing Xie
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, help="argument to decide if you are going to use Image Augmentation", type=bool)
    parser.add_argument("--mixup", action=argparse.BooleanOptionalAction, help="argument to decide if you are going to use Mixup", type=bool)
    parser.add_argument("--augmix", action=argparse.BooleanOptionalAction, help="argument to decide if you are going to use AugMix", type=bool)
    parser.add_argument("-p","--P", default=1.0, help="argument to decide the propability of using the Augmentation",type=float)
    parser.add_argument("--smart", action=argparse.BooleanOptionalAction, help="argument to decide if you are going to use SmartAugment", type=bool)
    parser.add_argument("--segment", default=0, help="Segmentation type: 0 is no segmentation, 1 is color segmentation, 2 is mix, 3 is blurred segmentation", type=int, choices=[0, 1, 2, 3])
    parser.add_argument("--model", help="Path to pretrained pix2pix GAN model", type=str)
    parser.add_argument("--gan", default="image2stylegan", type=str, help="gan type we used to generate images")
    parser.add_argument("--batchsize", default=32,type=int, help="batch size")
    parser.add_argument("--pretrain", action=argparse.BooleanOptionalAction, help="decide if you are going to use pretrained vgg model")
    parser.add_argument("--use_cuda", action='store', default=True, type=bool)
    parser.add_argument("--model_dir", action='store', default="pretrain_stylegan", type=str)
    parser.add_argument("--model_name", action='store', default="karras2019stylegan-ffhq-1024x1024.pt", type=str)
    parser.add_argument("--lr", action='store', default=0.015, type=float)
    parser.add_argument("--epochs", action='store', default=1000, type=int)
    parser.add_argument("--device", default='cuda:0', help="whether use gpu or not")
    parser.add_argument("--data_path", default='dataset/TSKinFace_Data/TSKinFace_cropped/', help="dataset path", type=str)

    # Argument to decide which dataset to use
    parser.add_argument("--dataset","-d",default="FMD", help="argument to decide which dataset to use. Default setting is FMD", type=str)

    # Parsing the argument to the args
    args = parser.parse_args()

    # Include some very basic sanity check here to make sure the codes does not crash inside main()
    ## TODO: Jiaqing Xie
    if args.P > 1.0 or args.P < 0.0:
        raise ValueError("The Probability for argument -p has to be between 0.0 and 1.0!!")

    if args.segment > 0 and args.model is None:
        raise ValueError("Please specify the model path for the pix2pix GAN")

    if not args.augment:
        if args.mixup or args.augmix or args.smart:
            raise ValueError("MixUp, AugMix or SmartAugment can only be used when you set the Augment flag!!!")
    
    if args.augment:
        if not args.mixup and not args.augmix and not args.smart:
            raise ValueError("One of MixUp, AugMix or SmartAugment has to be specified for your augmentation task!!!")
    
    if args.dataset not in ['FMD', 'FMSD', 'FMS']:
        raise ValueError("Invalid argument setting for dataset. Only FMD, FMS or FMSD accepted!!")

    main(args)