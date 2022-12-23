# This is the run script for the final pipeline version.
# Please make sure every dependency is installed in the requirements.txt file.

import argparse
import gc
import logging
import numpy as np
import torch
import torch.nn.functional as F
import os
from augmentations import augmentations as A
from augmentations.TSKinFace_Dataset import TSKinDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from ImageSegmentation.face_parsing.face_parsing_test import face_parsing_test
from ImageSegmentation.pix2pixGAN.test import test_pix2pix
from MLP2 import FourLayerNet
from torchsummary import summary

import matplotlib.pyplot as plt
import imageio

from PipelineGAN.stylegan_train import load, embedding_function, style_transfer
from PipelineGAN.utils import PSNR, loss_function
from PipelineGAN.VGG16 import VGG16_perceptual

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(args):

    # Branch when you decide to use the Image Augmentation.
    # MixUp
    if args.augment and args.mixup:
        print("Image Augmentation is being used. Method is MixUp. Probability is "+str(args.P))
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
        print("Image Augmentation is being used. Method is AugMix. Probability is "+str(args.P))
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
    
    print('Length of dataset ', len(dataset))
    # Use Image Segmentation.
    dataset_orig = dataset
    if args.segment:
        seg_path = args.data_path + args.dataset + '_seg_' + str(args.segment-1)
        if args.augment:
            seg_path = seg_path + '_aug-'
            if args.mixup:
                seg_path = seg_path + 'mixup/'
            else:
                seg_path = seg_path + 'augmix/'
        else:
            seg_path = seg_path + '/'

        if not args.load_seg:
            print("Image Segmentation is being applied to the parent images.")
            data_in = []
            label_list = []
            for images, labels in dataset:
                data_in.append(np.transpose(images.numpy(), (1, 2, 0)))
                label_list.append(labels)
            data_in = np.asarray(data_in)

            # Segment all images (father, mother, child)
            # print(len(data_in))
            data_out = face_parsing_test(input_images=data_in, blurring=args.segment-1, device=args.device)

            # Save images to separate segmentation folder
            father_path = seg_path + args.dataset + '-F/'
            mother_path = seg_path + args.dataset + '-M/'
            child_path = seg_path + args.dataset + '-Z/'
            if not os.path.exists(father_path):
                os.makedirs(father_path)
            if not os.path.exists(mother_path):
                os.makedirs(mother_path)
            if not os.path.exists(child_path):
                os.makedirs(child_path)
            f = 0
            m = 0
            l = len(data_out)
            if args.dataset == 'FMSD':
                l = l/4
            else:
                l = l/3
            delete_list = []
            for i, (im, label) in enumerate(list(zip(data_out, label_list))):
                if label == 0:
                    if np.any(im):
                        imageio.imwrite(father_path + args.dataset + "-{}-F.png".format(i + 1), im)
                        f += 1
                        m = f
                    else:
                        delete_list.append(i)
                        delete_list.append(i + l)
                        if args.dataset == 'FMSD':
                            delete_list.append(2*(i + l))
                            delete_list.append(2*(i + l) + 1)
                        else:
                            delete_list.append(i + 2*l)
                elif label == 1:
                    if np.any(im):
                        imageio.imwrite(mother_path + args.dataset + "-{}-M.png".format((i + 1)-f), im)
                        m += 1
                    else:
                        delete_list.append(i)
                        delete_list.append(i - l)
                        if args.dataset == 'FMSD':
                            delete_list.append(2*i)
                            delete_list.append(2*i + 1)
                        else:
                            delete_list.append(i + l)
                elif i%2 == 0 and label == 2 and args.dataset == 'FMSD':
                    if np.any(im):
                        imageio.imwrite(child_path + args.dataset + "-{}-D.png".format(int((i + 2 - m)/2)), im)
                    else:
                        delete_list.append(i)
                        delete_list.append(i+1)
                        delete_list.append(i/2)
                        delete_list.append(i/2 - l)
                elif i%2 == 1 and label == 2 and args.dataset == 'FMSD':
                    if np.any(im):
                        imageio.imwrite(child_path + args.dataset + "-{}-S.png".format(int((i + 1 - m)/2)), im)
                    else:
                        delete_list.append(i-1)
                        delete_list.append(i)
                        delete_list.append((i-1)/2)
                        delete_list.append((i-1)/2 - l)
                elif label == 2:
                    if np.any(im):
                        imageio.imwrite(child_path + args.dataset + "-{}-".format(i + 1 - m) + args.dataset[-1] + ".png", im)
                    else:
                        delete_list.append(i)
                        delete_list.append(i - l)
                        delete_list.append(i - 2*l)

        print("Image Segmentation Dataset is loaded.")

        dataset_orig = dataset
        dataset = ImageFolder(
            root = seg_path,
            transform = transforms.Compose([
                transforms.CenterCrop((1024, 1024)),
                transforms.ToTensor(),
            ])
        )

    path = 'temp/' + args.dataset
    result_path = 'result/' + args.dataset
    if args.segment != 0:
        path = path + '_seg' + str(args.segment-1)
        result_path = result_path + '_seg' + str(args.segment-1)
    if args.augment:
        path = path + '_aug-'
        result_path = result_path + '_aug-'
        if args.mixup:
            path = path + 'mixup/'
            result_path = result_path + 'mixup/'
        else:
            path = path + 'augmix/'
            result_path = result_path + 'augmix/'
    else:
        path = path + '/'
        result_path = result_path + '/'

    if not os.path.exists(path):
        print('Creating directory')
        os.makedirs(path)
    if not os.path.exists(path + 'gan_reconstr/'):
        os.makedirs(path + 'gan_reconstr/')
    if not os.path.exists(path + 'gan_latent/'):
        os.makedirs(path + 'gan_latent/')

    # GAN projection into latent space
    # TODO: Jiaqing Xie
    print('Converting parent images to latent space with Image2StyleGAN')
    latent_f = []
    latent_m = []
    latent_c = []

    _, g_synthesis = load(args)

    idx = 0
    c_idx = []
    f_idx = []
    m_idx = []

    for image, label in dataset:
        if label == 0:
            f_idx.append(idx)
            if args.dataset == 'FMSD':
                f_idx.append(idx)
        elif label == 1:
            m_idx.append(idx)
            if args.dataset == 'FMSD':
                m_idx.append(idx)
        elif label == 2:
            c_idx.append(idx)
        idx+=1

    for num, (f, m, c) in enumerate(list(zip(f_idx,m_idx, c_idx))):
        print('Embedding parents and children triplets, number', num+1)
        # father image
        image_f = dataset[f][0]
        image_f = image_f[None, :]
        image_f = image_f.to(args.device)

        # mother image
        image_m = dataset[m][0]
        image_m = image_m[None, :]
        image_m = image_m.to(args.device)

        image_c = dataset[c][0]
        image_c = image_c[None, :]
        image_c = image_c.to(args.device)
        
        p = path + "gan_latent/reconstruct_{}_".format(num + 1)
        lf = embedding_function(image_f, args, g_synthesis, args.epochs_lat, p + "f_").to(args.device)
        lm = embedding_function(image_m, args, g_synthesis, args.epochs_lat, p + "m_").to(args.device)
        lc = embedding_function(image_c, args, g_synthesis, args.epochs_lat, p + "c_").to(args.device)

        latent_f.append(lf)
        latent_m.append(lm)
        latent_c.append(lc)
    
    torch.cuda.empty_cache()

    # Feature selection
    print('Setting up feature selection.')
    model_p = FourLayerNet()
    print(summary(model_p, torch.cat((latent_f[0], latent_m[0]), dim=2).shape))
    model_p = model_p.to(args.device)
    optim = torch.optim.Adam(list(model_p.parameters()),  lr=args.lr)
    
    print('Starting training')
    for epoch in range(args.epochs):
        print('Epoch', epoch + 1)
        for num, (lf, lm, lc) in enumerate(zip(latent_f, latent_m, latent_c)):
            model_p.train()
            optim.zero_grad()
            latent_p = torch.cat((lf, lm), dim=2)
            child_pred = model_p(latent_p)
            child_pred = torch.reshape(child_pred[0], (18,512))[None, :]
            syn_child_img = g_synthesis(child_pred)
            syn_child_img = ((syn_child_img + 1.0) / 2.0).clamp(0, 1)
            loss = F.mse_loss(input=syn_child_img, target=dataset[num][0], reduction='mean')
            loss.backward()  # backpropagation loss
            optim.step()
            if (epoch+1) % 2 == 0:
                print("iter{}: , mse_loss --{}".format(epoch + 1, loss.item()))
                save_image(syn_child_img.clamp(0, 1), path + "gan_reconstr/reconstruct_{}_{}.png".format(epoch + 1, num))    

    # Undo image segmentation
    # TODO: Sofie Daniëls
    data_list_seg_gen = []
    data_list_real = []
    print('Removing center crop.')
    for i, (f, m) in enumerate(zip(latent_f, latent_m)):
        print('syn child img', i)
        latent_p = torch.cat((f, m), dim=2)
        child_pred = model_p(latent_p)
        child_pred = torch.reshape(child_pred[0], (18,512))[None, :]
        syn_child_img = g_synthesis(child_pred)
        syn_child_img = ((syn_child_img + 1.0) / 2.0).clamp(0, 1)
        data_list_seg_gen.append(np.transpose(syn_child_img.detach().cpu().numpy(), (1, 2, 0))*255)
        nim = np.transpose(dataset_orig[c_idx[i]][0].numpy(), (1, 2, 0))*255
        #plt.imshow(nim/255)
        #plt.show()
        black_row = ~np.all(np.all(nim==0, axis=1), axis=1)
        black_col = ~np.all(np.all(nim==0, axis=0), axis=1)
        nim = nim[black_row,:,:] #remove all black rows
        nim = nim[:,black_col,:] #remove all black columns
        #plt.imshow(nim/255)
        #plt.show()
        data_list_real.append(nim)
        torch.cuda.empty_cache()
        del syn_child_img
        del new_latent_f
        del new_latent_m
        del latent
        gc.collect()
        torch.cuda.empty_cache()
    
    if args.segment:
        print('Undoing segmentation.')
        #plt.imshow(data_list_seg_gen[0]/255)
        #plt.show()
        if not os.path.exists(path + 'pix2pix_out/'):
            os.makedirs(path + 'pix2pix_out/')
        data_list_real_gen = test_pix2pix([data_list_seg_gen, data_list_real], args.model, path + 'pix2pix_out/')
    else:
        data_list_real_gen = data_list_seg_gen
        
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print('Saving results.')
    for i, im in enumerate(data_list_real_gen):
        if args.dataset == 'FMSD':
            if i%2 == 0:
                imageio.imwrite(result_path + args.dataset + "-{}-D.png".format(int((i + 2)/2)), im)
            else:
                imageio.imwrite(result_path + args.dataset + "-{}-S.png".format(int((i + 1)/2)), im)
        else:
            imageio.imwrite(result_path + args.dataset + "-{}-".format(i + 1) + args.dataset[-1] + ".png", im)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Include the arguments you want to experiment here
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, type=bool,
                        help="argument to decide if you are going to use Image Augmentation")
    parser.add_argument("--mixup", action=argparse.BooleanOptionalAction, type=bool,
                        help="argument to decide if you are going to use Mixup")
    parser.add_argument("--augmix", action=argparse.BooleanOptionalAction, type=bool,
                        help="argument to decide if you are going to use AugMix")
    parser.add_argument("--P", "-p", default=1.0, type=float,
                        help="argument to decide the propability of using the Augmentation")
    parser.add_argument("--smart", action=argparse.BooleanOptionalAction, type=bool,
                        help="argument to decide if you are going to use SmartAugment")

    parser.add_argument("--segment", default=0, type=int, choices=[0, 1, 2, 3],
                        help="Segmentation type: 0 is no segmentation, 1 is color segmentation, 2 is mix, 3 is blurred segmentation")
    parser.add_argument("--model", type=str, help="Path to pretrained pix2pix GAN model")
    parser.add_argument("--load-seg", default=False, type=bool,
                        help="Whether to load previous segmentations or start anew")
    
    parser.add_argument("--gan", default="image2stylegan", type=str,
                        help="gan type we used to generate images")
    parser.add_argument("--batchsize", default=32, type=int, help="batch size")
    parser.add_argument("--pretrain", action=argparse.BooleanOptionalAction,
                        help="decide if you are going to use pretrained vgg model")
    parser.add_argument("--use_cuda", action='store', default=True, type=bool)
    parser.add_argument("--model_dir", action='store', default="pretrain_stylegan", type=str)
    parser.add_argument("--model_name", action='store',
                        default="karras2019stylegan-ffhq-1024x1024.pt", type=str)
    parser.add_argument("--lr", action='store', default=0.015, type=float)
    parser.add_argument("--epochs", action='store', default=6, type=int)
    parser.add_argument("--epochs_lat", action='store', default=2, type=int)
    parser.add_argument("--device", default='cuda:0', help="Whether to use a GPU or not")

    parser.add_argument("--data_path", default='dataset/TSKinFace_Data_HR/TSKinFace_cropped/',
                        type=str, help="dataset path")

    # Argument to decide which dataset to use
    parser.add_argument("--dataset","-d",default="FMD", type=str,
                        help="argument to decide which dataset to use. Default setting is FMD")

    # Parsing the argument to the args
    args = parser.parse_args()

    # Include some very basic sanity check here to make sure the codes does not crash inside main()
    ## TODO: Jiaqing Xie
    if args.P > 1.0 or args.P < 0.0:
        raise ValueError("The Probability for argument -p has to be between 0.0 and 1.0!")
    
    if not args.augment:
        if args.mixup or args.augmix or args.smart:
            raise ValueError("MixUp, AugMix or SmartAugment can only be used when you set the Augment flag!")
    
    if args.augment:
        if not args.mixup and not args.augmix and not args.smart:
            raise ValueError("One of MixUp, AugMix or SmartAugment has to be specified for your augmentation task!")
    
    if args.dataset not in ['FMD', 'FMSD', 'FMS']:
        raise ValueError("Invalid argument setting for dataset. Only FMD, FMS or FMSD accepted!")

    if args.segment > 0 and args.model is None:
        raise ValueError("Please specify the model path for the pix2pix GAN")

    main(args)
