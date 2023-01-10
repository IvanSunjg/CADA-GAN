# This is the run script for the final pipeline version.
# Please make sure every dependency is installed in the requirements.txt file.

import argparse
import gc
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from augmentations import augmentations as A
from augmentations.TSKinFace_Dataset import TSKinDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from ImageSegmentation.face_parsing.face_parsing_test import face_parsing_test
from ImageSegmentation.pix2pixGAN.test import test_pix2pix
from torchsummary import summary
from scipy.spatial import distance
from skimage.transform import resize

import matplotlib.pyplot as plt
import imageio

from projector import run_projection
from generate import generate_images

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class WeightLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor([0.5]).to(args.device))

    def forward(self, lf, lm):
        # lf, lm must be in the same device with gamma
        lat = lf * self.gamma + lm * (1 - self.gamma)
        return lat

def main(args):
    logging.info("Setting up logger")

    # Branch when you decide to use the Image Augmentation.
    if args.augment:
        # MixUp
        if args.mixup:
            logging.info("Image Augmentation is being used. Method is MixUp. Probability is " + str(args.P))
            default_dataset = ImageFolder(
                root = args.data_path + args.dataset+'/',
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                ])
            )
            if args.dataset == 'FMD':
                dataset = TSKinDataset(
                    root = args.data_path + args.dataset+'/',
                    transform = [
                        transforms.ToTensor(),
                        transforms.Resize((512, 512)),
                        A.P(A.MixUp_FMD(dataset=default_dataset), p=args.P)
                    ]
                )
            elif args.dataset == 'FMS':
                dataset = TSKinDataset(
                    root = args.data_path + args.dataset+'/',
                    transform = [
                        transforms.ToTensor(),
                        transforms.Resize((512, 512)),
                        A.P(A.MixUp_FMS(dataset=default_dataset), p=args.P)
                    ]
                )
            elif args.dataset == 'FMSD':
                dataset = TSKinDataset(
                    root = args.data_path + args.dataset+'/',
                    transform = [
                        transforms.ToTensor(),
                        transforms.Resize((512, 512)),
                        A.P(A.MixUp_FMSD(dataset=default_dataset), p=args.P)
                    ]
                )
        # AugMix
        elif args.augmix:
            logging.info("Image Augmentation is being used. Method is AugMix. Probability is " + str(args.P))
            dataset = TSKinDataset(
                root = args.data_path + args.dataset+'/',
                transform = [
                    transforms.ToTensor(),
                    transforms.Resize((512, 512)),
                    A.P(A.AugMix(), p=args.P)
                ]
            )
        else:
            dataset = ImageFolder(
                root = args.data_path + args.dataset+'/',
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                ])
            )
    # Basic Unaugmented Dataset 
    else:
        dataset = ImageFolder(
            root = args.data_path + args.dataset+'/',
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
        )
    
    logging.info('Length of dataset ' + str(len(dataset)))
    # Use Image Segmentation.
    dataset_orig = dataset
    if args.segment:
        # Set up path for saving/loading segmented images
        seg_path = args.data_path + args.dataset + '_seg_' + str(args.segment-1)
        if args.augment:
            seg_path = seg_path + '_aug-'
            if args.mixup:
                seg_path = seg_path + 'mixup/'
            else:
                seg_path = seg_path + 'augmix/'
        else:
            seg_path = seg_path + '/'

        # Segment from scratch if no segmented images are made available
        if not args.load_seg:
            logging.info("Image Segmentation is being applied to the parent images.")
            data_in = []
            label_list = []
            # Loop to get numpy arrays from Tensors
            for images, labels in dataset:
                # Transpose to account for tensor to numpy change
                data_in.append(np.transpose(images.numpy(), (1, 2, 0)))
                label_list.append(labels)
            data_in = np.asarray(data_in)

            # Segment all images (father, mother, child)
            data_out = face_parsing_test(input_images=data_in, blurring=args.segment-1, device=args.device)

            # Save images to separate segmentation folder (create if non-existing)
            father_path = seg_path + args.dataset + '-F/'
            mother_path = seg_path + args.dataset + '-M/'
            child_path = seg_path + args.dataset + '-Z/'
            if not os.path.exists(father_path):
                os.makedirs(father_path)
            if not os.path.exists(mother_path):
                os.makedirs(mother_path)
            if not os.path.exists(child_path):
                os.makedirs(child_path)
            
            
            #TODO CODE TO DELETE TRIPLETS WHEN FACE PARSING FAILS
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
            
        logging.info("Image Segmentation Dataset is loaded.")

        # Save original dataset in 'dataset_orig', and new one in 'dataset'.
        dataset_orig = dataset
        dataset = ImageFolder(
            root = seg_path,
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
        )

    # Create folders and paths for temporary results from GAN
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
        logging.info('Creating directory')
        os.makedirs(path)
    if not os.path.exists(path + 'gan_reconstr/'):
        os.makedirs(path + 'gan_reconstr/')
    if not os.path.exists(path + 'gan_latent/'):
        os.makedirs(path + 'gan_latent/')

    # GAN projection into latent space with styleGAN2 code
    logging.info('Converting parent and child images to latent space with StyleGAN2')
    idx = 0
    c_idx = []
    f_idx = []
    m_idx = []

    # Loop through dataset to get indices of images.
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

    im_f = []
    im_m = []
    im_c = []

    # Loop through dataset to get images using indices from previous for loop
    for num, (f, m, c) in enumerate(list(zip(f_idx,m_idx, c_idx))):
        logging.info('Embedding parents and children triplets, number ' + str(num+1))
        # father image
        image_f = dataset[f][0]

        # mother image
        image_m = dataset[m][0]

        # child image
        image_c = dataset[c][0]

        im_f.append(image_f)
        im_m.append(image_m)
        im_c.append(image_c)

    latent_f = []
    latent_m = []
    latent_c = []

    # TODO PRETRAIN MODEL PARTIALLY
    # Convert images to latent vectors with StyleGAN2
    latent_f, latent_m, latent_c = run_projection(network_pkl=args.stylegan_model,
                                                  t_father=im_f, t_mother=im_m, t_child=im_c,
                                                  outdir=path + "gan_latent", seed=303,
                                                  num_steps=args.epochs_lat, save_video=False)
    
    # Feature selection with 4-layer network
    logging.info('Setting up feature selection.')
    
    # Index where to split for train/test
    train_test_split = int(round(args.ratio*len(f_idx)))-1
    logging.info('Traintestsplit ' + str(train_test_split))

    WLModel = WeightLatent()
    optim = torch.optim.Adam(list(WLModel.parameters()), lr=0.01)
    

    logging.info('Starting training')
    for epoch in range(args.epochs):
        logging.info('Epoch ' + str(epoch + 1))
        #logging.info('Ziplatent length ' + str(len(list(zip(latent_f, latent_m, latent_c)))))
        for num, (lf, lm, lc) in enumerate(list(zip(latent_f, latent_m, latent_c))[0:train_test_split]):
            WLModel.train()
            optim.zero_grad()
            lf_n = np.load(lf)
            lf_n = lf_n['w']
            lf_n = torch.tensor(lf_n).to(args.device)
            lm_n = np.load(lm)
            lm_n = lm_n['w']
            lm_n = torch.tensor(lm_n).to(args.device)
            lc_n = np.load(lc)
            lc_n = lc_n['w']
            lc_n = torch.tensor(lc_n).to(args.device)
            #logging.info('latent f shape ' + str(lf_n.shape))
            # Predict child latent vector
            child_pred = WLModel(lf_n, lm_n)
            #logging.info('child pred shape ' + str(child_pred.shape))
            child_pred = child_pred.to(args.device)
            #logging.info('child pred shape ' + str(child_pred.shape))
            # Update loss depending on similarity of predicted child latent vector with real child latent vector
            # TODO UPDATE LOSS TO IMAGE INSTEAD OF LAT VECTOR
            np.savez(f'{path + "gan_latent"}/projected_w_temp_' + "{}_{}.npz".format(epoch, num), w=child_pred.detach().cpu().numpy())
            [child_pred_im, child_or_im] = generate_images(network_pkl=args.stylegan_model, outdir=path + "gan_latent", projected_w=[f'{path + "gan_latent"}/projected_w_temp_' + "{}_{}.npz".format(epoch, num), lc])
            loss_im = F.mse_loss(input=torch.from_numpy(child_pred_im)[None, :].to(torch.float).to(args.device), target=torch.from_numpy(child_or_im)[None, :].to(torch.float).to(args.device), reduction='mean')
            loss = F.mse_loss(input=child_pred, target=lc_n, reduction='mean')
            loss.backward()
            optim.step()
            logging.info("iter{}: , mse_loss --{}, mse_loss_im --{}".format(epoch + 1, loss.item(), loss_im))
            
    data_list_seg_gen = []
    data_list_real = []
    logging.info('Starting evaluation.')

    # TODO SPECIFY GENDER IN ADVANCE FOR FMSD
    # Test trained trainable weight a and obtain predicted latent vectors of children 
    print("trained weight: {}".format(WLModel.gamma))
    lat_saves = []
    for i, (lf, lm) in enumerate(list(zip(latent_f[train_test_split:], latent_m[train_test_split:]))):
        j = i + train_test_split
        WLModel.eval()
        lf_n = np.load(lf)
        lf_n = lf_n['w']
        lf_n = torch.tensor(lf_n).to(args.device)
        lm_n = np.load(lm)
        lm_n = lm_n['w']
        lm_n = torch.tensor(lm_n).to(args.device)
        lc_n = np.load(lc)
        lc_n = lc_n['w']
        lc_n = torch.tensor(lc_n).to(args.device)

        child_pred = WLModel(lf_n, lm_n)
        child_pred = torch.reshape(child_pred[0], (16,512))[None, :]
        #logging.info('child pred final shape ' + str(child_pred.shape))
        np.savez(f'{path + "gan_latent"}/projected_w_' + "{}.npz".format(j), w=child_pred.detach().cpu().numpy())
        lat_saves.append(f'{path + "gan_latent"}/projected_w_' + "{}.npz".format(j))

    data_list_seg_gen = generate_images(network_pkl=args.stylegan_model, outdir=path + "gan_reconstr", projected_w=lat_saves)
    #logging.info('datalistseggen len ' + str(len(data_list_seg_gen)))
    #logging.info('datalistseggen entry shape ' + str(data_list_seg_gen[0].shape))
    for i in range(train_test_split, len(latent_f)):
        print(train_test_split, len(latent_f))
        or_im = np.transpose(dataset_orig[c_idx[i]][0].numpy(), (1, 2, 0))
        #plt.imshow(or_im)
        #plt.show()
        data_list_real.append(or_im)
    
    #logging.info('datalistreal len ' + str(len(data_list_real)))
    #logging.info('datalistreal entry shape ' + str(data_list_real[0].shape))

    # Undo image segmentation with pretrained pix2pix GAN
    if args.segment:
        logging.info('Undoing segmentation.')
        #plt.imshow(data_list_seg_gen[0])
        #plt.show()
        if not os.path.exists(path + 'pix2pix_out/'):
            os.makedirs(path + 'pix2pix_out/')
        data_list_real_gen = test_pix2pix([data_list_seg_gen, data_list_real], args.model, path + 'pix2pix_out/')
    else:
        data_list_real_gen = data_list_seg_gen
    
    # TODO PLOT EVALUATION (COSINE SIMILARITY)
    # Save final images
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    logging.info('Saving results.')
    t_csm = 0
    for i, im in enumerate(data_list_real_gen):
        j = train_test_split + i
        r_path = result_path + args.dataset
        if args.dataset == 'FMSD':
            if j%2 == 0:
                r_path = r_path + "-{}-D".format(int((j + 2)/2))
                imageio.imwrite(r_path + ".png", im.astype(np.uint8))
            else:
                r_path = r_path + "-{}-S".format(int((j + 1)/2))
                imageio.imwrite(r_path + ".png", im.astype(np.uint8))
        else:
            r_path = r_path + "-{}-".format(j + 1) + args.dataset[-1]
            imageio.imwrite(r_path + ".png", im.astype(np.uint8))
        im_or = resize(data_list_real[i], im.shape, anti_aliasing=True)
        csm = distance.cosine(im.flatten(), im_or.flatten())
        logging.info("Cosine Similarity for image " + str(j) + " --" + str(csm))
        t_csm += csm

        # plot father image
        plt.subplot(3, 3, 1)
        plt.axis('off')
        plt.imshow(np.transpose(im_f[j].numpy(), (1, 2, 0)))
        # plot mother image
        plt.subplot(3, 3, 3)
        plt.axis('off')
        plt.imshow(np.transpose(im_m[j].numpy(), (1, 2, 0)))
        # plot predicted child image
        plt.subplot(3, 3, 5)
        plt.axis('off')
        plt.imshow(im)
        # plot real child face
        plt.subplot(3, 3, 8)
        plt.axis('off')
        plt.imshow(im_or)
        # save plot to file
        filename = r_path + "_plot.png"
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.0,
                            hspace=0.0)
        plt.savefig(filename, dpi=500)
        plt.close()

    logging.info("Cosine Similarity Total Average --" + str(t_csm/(len(latent_f)-train_test_split)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", type=bool, default=False,
                        help="argument to decide if you are going to use Image Augmentation")
    parser.add_argument("--mixup", type=bool, default=False,
                        help="argument to decide if you are going to use Mixup")
    parser.add_argument("--augmix", type=bool, default=False,
                        help="argument to decide if you are going to use AugMix")
    parser.add_argument("--P", "-p", default=1.0, type=float,
                        help="argument to decide the propability of using the Augmentation")
    parser.add_argument("--segment", default=0, type=int, choices=[0, 1, 2, 3],
                        help="segmentation type: 0 is no segmentation, 1 is color segmentation, 2 is mix, 3 is blurred segmentation")
    parser.add_argument("--model", type=str, help="path to pretrained pix2pix GAN model")
    parser.add_argument("--load-seg", default=False, type=bool,
                        help="whether to load previous segmentations or start anew")
    parser.add_argument("--ratio", default=0.8, type=float, help="train to test ratio")
    parser.add_argument("--stylegan-model", action='store',
                        default="pretrained/ffhq-512-avg-tpurun1.pkl", type=str,
                        help="location of pretrained StyleGAN2 model")
    parser.add_argument("--lr", action='store', default=0.015, type=float,
                        help="learning rate for parent latent vector mixing")
    parser.add_argument("--epochs", action='store', default=100, type=int,
                        help="number of epochs to train parent latent vectors mixing")
    parser.add_argument("--epochs-lat", action='store', default=850, type=int,
                        help="number of epochs to run latent vector generator")
    parser.add_argument("--device", default='cuda:0', help="cuda device (gpu only)")
    parser.add_argument("--data-path", default='dataset/TSKinFace_Data_HR/TSKinFace_cropped/',
                        type=str, help="path to dataset")
    parser.add_argument("--dataset","-d",default="FMD", type=str,
                        help="argument to decide which dataset to use. Default setting is FMD")

    # Parsing the argument to the args
    args = parser.parse_args()

    # Include some very basic sanity check here to make sure the codes does not crash inside main()
    if args.P > 1.0 or args.P < 0.0:
        raise ValueError("The Probability for argument -p has to be between 0.0 and 1.0!")
    
    if not args.augment:
        if args.mixup or args.augmix:
            raise ValueError("MixUp and AugMix can only be used when you set the Augment flag!")
    
    if args.augment:
        if not args.mixup and not args.augmix:
            raise ValueError("Either MixUp or AugMix has to be specified for your augmentation task!")
    
    if args.dataset not in ['FMD', 'FMSD', 'FMS']:
        raise ValueError("Invalid argument setting for dataset. Only FMD, FMS or FMSD accepted!")

    if args.segment > 0 and args.model is None:
        raise ValueError("Please specify the model path for the pix2pix GAN.")
    
    if args.ratio <= 0.0 or args.ratio >= 1.0:
        raise ValueError("Please account for both training and testing and specify the ratio accordingly.")

    logging.basicConfig(filename='log.txt',
                        filemode='a',
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    main(args)
