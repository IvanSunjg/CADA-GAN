from .stylegan_layers import G_mapping,G_synthesis
from .VGG16 import VGG16_perceptual
from .utils import image_reader, loss_function, PSNR, get_device
import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import os


def load(args):

    device = get_device(args.use_cuda)
    g_all = nn.Sequential(OrderedDict([('g_mapping', G_mapping()),
                                    ('g_synthesis', G_synthesis(resolution=1024))
                                    ]))

    # Load the pre-trained model
    g_all.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name), map_location=device))
    g_all.eval()
    g_all.to(device)
    g_mapping, g_synthesis = g_all[0], g_all[1]

    return args, g_synthesis


def embedding_function(image, args, g_synthesis):
    upsample = torch.nn.Upsample(scale_factor=256 / 1024, mode='bilinear')
    img_p = image.clone()
    img_p = upsample(img_p)
    perceptual = VGG16_perceptual().to(args.device)

    MSE_loss = nn.MSELoss(reduction="mean")
    latents = torch.zeros((1, 18, 512), requires_grad=True, device=args.device)
    optimizer = optim.Adam({latents}, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    loss_ = []
    loss_psnr = []
    for e in range(args.epochs):
        optimizer.zero_grad()
        syn_img = g_synthesis(latents)
        syn_img = (syn_img + 1.0) / 2.0
        mse, per_loss = loss_function(syn_img, image, img_p, MSE_loss, upsample, perceptual)
        psnr = PSNR(mse, flag=0)
        loss = per_loss + mse
        loss.backward()
        optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        loss_p = per_loss.detach().cpu().numpy()
        loss_m = mse.detach().cpu().numpy()
        loss_psnr.append(psnr)
        loss_.append(loss_np)
        if (e + 1) % 500 == 0:
            print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(e + 1, loss_np, loss_m,
                                                                                            loss_p, psnr))
            #save_image(syn_img.clamp(0, 1), "save_images/reconstruct_{}.png".format(e + 1))

    # plt.plot(loss_, label='Loss = MSELoss + Perceptual')
    # plt.plot(loss_psnr, label='PSNR')
    # plt.legend()
    return latents


def style_transfer(target_latent, style_latent, src, tgt, g_synthesis):
    '''
        style transfer
    '''
    tmp_latent1 = target_latent[:, :10, :]
    tmp_latent2 = style_latent[:, 10:, :]
    latent = torch.cat((tmp_latent1, tmp_latent2), dim=1)
    print(latent.shape)
    syn_img = g_synthesis(latent)
    syn_img = (syn_img + 1.0) / 2.0
    save_image(syn_img.clamp(0, 1), "Style_transfer_{}_{}_10.png".format(src, tgt))