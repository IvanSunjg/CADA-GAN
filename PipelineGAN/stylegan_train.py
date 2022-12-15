from stylegan_layers import G_mapping,G_synthesis
from VGG16 import VGG16_perceptual
from utils import image_reader, loss_function, PSNR, get_device
import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--use_cuda", action='store', default=True, type=bool)
parser.add_argument("--model_dir", action='store', default="pretrain_stylegan", type=str)
parser.add_argument("--model_name", action='store', default="karras2019stylegan-ffhq-1024x1024.pt", type=str)
parser.add_argument("--images_dir", action='store', default="images/image2stylegan", type=str)
parser.add_argument("--lr", action='store', default=0.01, type=float)
parser.add_argument("--epochs", action='store', default=1500, type=int)
args = parser.parse_args()

device = get_device(args.use_cuda)
g_all = nn.Sequential(OrderedDict([('g_mapping', G_mapping()),
                                   ('g_synthesis', G_synthesis(resolution=1024))
                                   ]))

# Load the pre-trained model
g_all.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name), map_location=device))
g_all.eval()
g_all.to(device)
g_mapping, g_synthesis = g_all[0], g_all[1]

print("success")


def embedding_function(image):
    upsample = torch.nn.Upsample(scale_factor=256 / 1024, mode='bilinear')
    img_p = image.clone()
    img_p = upsample(img_p)
    perceptual = VGG16_perceptual().to(device)

    MSE_loss = nn.MSELoss(reduction="mean")
    latents = torch.zeros((1, 18, 512), requires_grad=True, device=device)
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
            save_image(syn_img.clamp(0, 1), "save_images/reconstruct_{}.png".format(e + 1))

    plt.plot(loss_, label='Loss = MSELoss + Perceptual')
    plt.plot(loss_psnr, label='PSNR')
    plt.legend()
    return latents


def morphing(w0, w1, img_id0, img_id1):
    '''
        morphing operation
    '''
    for i in range(20):
        a = (1 / 20) * i
        w = w0 * (1 - a) + w1 * a
        syn_img = g_synthesis(w)
        syn_img = (syn_img + 1.0) / 2.0
        save_image(syn_img.clamp(0, 1), "save_images/image2stylegan/morphing/morphed_{}_{}_{}.png".format(img_id0, img_id1, i))


def style_transfer(target_latent, style_latent, src, tgt):
    '''
        style transfer
    '''
    tmp_latent1 = target_latent[:, :10, :]
    tmp_latent2 = style_latent[:, 10:, :]
    latent = torch.cat((tmp_latent1, tmp_latent2), dim=1)
    print(latent.shape)
    syn_img = g_synthesis(latent)
    syn_img = (syn_img + 1.0) / 2.0
    save_image(syn_img.clamp(0, 1), "save_images/image2stylegan/style_transfer/Style_transfer_{}_{}_10.png".format(src, tgt))


img_path = os.path.join(args.images_dir, "geeks.jpg")
image0 = image_reader(img_path)
image0 = image0.to(device)


latent0 = embedding_function(image0)
print(latent0.shape)


# morphing(latent4, latent5, 4, 5)
# style_transfer(latent0, latent7, 0, 7)
# print("finished")
