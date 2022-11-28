---
title: StyleGAN 2
summary: >
 An annotated PyTorch implementation of StyleGAN2.
---

# StyleGAN 2

This is a [PyTorch](https://pytorch.org) implementation of the paper
 [Analyzing and Improving the Image Quality of StyleGAN](https://papers.labml.ai/paper/1912.04958)
 which introduces **StyleGAN 2**.
StyleGAN 2 is an improvement over **StyleGAN** from the paper
 [A Style-Based Generator Architecture for Generative Adversarial Networks](https://papers.labml.ai/paper/1812.04948).
And StyleGAN is based on **Progressive GAN** from the paper
 [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://papers.labml.ai/paper/1710.10196).
All three papers are from the same authors from [NVIDIA AI](https://twitter.com/NVIDIAAI).

*Our implementation is a minimalistic StyleGAN 2 model training code.
Only single GPU training is supported to keep the implementation simple.
We managed to shrink it to keep it at less than 500 lines of code, including the training loop.*

**🏃 Here's the training code: [`experiment.py`](experiment.html).**

![Generated Images](generated_64.png)

---*These are $64 \times 64$ images generated after training for about 80K steps.*---


We'll first introduce the three papers at a high level.

## Generative Adversarial Networks

Generative adversarial networks have two components; the generator and the discriminator.
The generator network takes a random latent vector ($z \in \mathcal{Z}$)
 and tries to generate a realistic image.
The discriminator network tries to differentiate the real images from generated images.
When we train the two networks together the generator starts generating images indistinguishable from real images.

## Progressive GAN

Progressive GAN generates high-resolution images ($1080 \times 1080$) of size.
It does so by *progressively* increasing the image size.
First, it trains a network that produces a $4 \times 4$ image, then $8 \times 8$ ,
 then an $16 \times 16$  image, and so on up to the desired image resolution.

At each resolution, the generator network produces an image in latent space which is converted into RGB,
with a $1 \times 1$  convolution.
When we progress from a lower resolution to a higher resolution
 (say from $4 \times 4$  to $8 \times 8$ ) we scale the latent image by $2\times$
 and add a new block (two $3 \times 3$  convolution layers)
 and a new $1 \times 1$  layer to get RGB.
The transition is done smoothly by adding a residual connection to
 the $2\times$ scaled $4 \times 4$  RGB image.
The weight of this residual connection is slowly reduced, to let the new block take over.

The discriminator is a mirror image of the generator network.
The progressive growth of the discriminator is done similarly.

![progressive_gan.svg](progressive_gan.svg)

---*$2\times$ and $0.5\times$ denote feature map resolution scaling and scaling.
$4\times4$, $8\times4$, ... denote feature map resolution at the generator or discriminator block.
Each discriminator and generator block consists of 2 convolution layers with leaky ReLU activations.*---

They use **minibatch standard deviation** to increase variation and
 **equalized learning rate** which we discussed below in the implementation.
They also use **pixel-wise normalization** where at each pixel the feature vector is normalized.
They apply this to all the convolution layer outputs (except RGB).


## StyleGAN

StyleGAN improves the generator of Progressive GAN keeping the discriminator architecture the same.

#### Mapping Network

It maps the random latent vector ($z \in \mathcal{Z}$)
 into a different latent space ($w \in \mathcal{W}$),
 with an 8-layer neural network.
This gives an intermediate latent space $\mathcal{W}$
where the factors of variations are more linear (disentangled).

#### AdaIN

Then $w$ is transformed into two vectors (**styles**) per layer,
 $i$, $y_i = (y_{s,i}, y_{b,i}) = f_{A_i}(w)$ and used for scaling and shifting (biasing)
 in each layer with $\text{AdaIN}$ operator (normalize and scale):
$$\text{AdaIN}(x_i, y_i) = y_{s, i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$

#### Style Mixing

To prevent the generator from assuming adjacent styles are correlated,
 they randomly use different styles for different blocks.
That is, they sample two latent vectors $(z_1, z_2)$ and corresponding $(w_1, w_2)$ and
 use $w_1$ based styles for some blocks and $w_2$ based styles for some blacks randomly.

#### Stochastic Variation

Noise is made available to each block which helps the generator create more realistic images.
Noise is scaled per channel by a learned weight.

#### Bilinear Up and Down Sampling

All the up and down-sampling operations are accompanied by bilinear smoothing.

![style_gan.svg](style_gan.svg)

---*$A$ denotes a linear layer.
$B$ denotes a broadcast and scaling operation (noise is a single channel).
StyleGAN also uses progressive growing like Progressive GAN.*---

## StyleGAN 2

StyleGAN 2 changes both the generator and the discriminator of StyleGAN.

#### Weight Modulation and Demodulation

They remove the $\text{AdaIN}$ operator and replace it with
 the weight modulation and demodulation step.
This is supposed to improve what they call droplet artifacts that are present in generated images,
 which are caused by the normalization in $\text{AdaIN}$ operator.
Style vector per layer is calculated from $w_i \in \mathcal{W}$ as $s_i = f_{A_i}(w_i)$.

Then the convolution weights $w$ are modulated as follows.
($w$ here on refers to weights not intermediate latent space,
 we are sticking to the same notation as the paper.)

$$w'_{i, j, k} = s_i \cdot w_{i, j, k}$$
Then it's demodulated by normalizing,
$$w''_{i,j,k} = \frac{w'_{i,j,k}}{\sqrt{\sum_{i,k}{w'_{i, j, k}}^2 + \epsilon}}$$
where $i$ is the input channel, $j$ is the output channel, and $k$ is the kernel index.

#### Path Length Regularization

Path length regularization encourages a fixed-size step in $\mathcal{W}$ to result in a non-zero,
 fixed-magnitude change in the generated image.

#### No Progressive Growing

StyleGAN2 uses residual connections (with down-sampling) in the discriminator and skip connections
 in the generator with up-sampling
  (the RGB outputs from each layer are added - no residual connections in feature maps).
They show that with experiments that the contribution of low-resolution layers is higher
 at beginning of the training and then high-resolution layers take over.