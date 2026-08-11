"""
image_gen/gan_model.py

A DCGAN (Deep Convolutional GAN) built from scratch in PyTorch.
Generator: random noise vector -> fake image
Discriminator: image -> real/fake probability

They train against each other: the Generator tries to fool the
Discriminator, the Discriminator tries not to be fooled. Over time the
Generator learns to produce increasingly realistic images.

This is sized for small images (28x28 or 32x32, grayscale or RGB) -
appropriate for Fashion-MNIST/MNIST-style datasets or small Kaggle
image datasets. Runs fine on a free Colab GPU (T4).
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1, feature_maps=64):
        super().__init__()
        self.latent_dim = latent_dim

        # Upsample from a latent noise vector to a full image via
        # transposed convolutions ("deconvolutions")
        self.net = nn.Sequential(
            # input: (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # state: (feature_maps*4, 4, 4)

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # state: (feature_maps*2, 8, 8)

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # state: (feature_maps, 16, 16)

            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # output: (img_channels, 32, 32), pixel values in [-1, 1]
        )

    def forward(self, z):
        # z shape: (batch, latent_dim) -> reshape to (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_maps=64):
        super().__init__()
        self.net = nn.Sequential(
            # input: (img_channels, 32, 32)
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (feature_maps, 16, 16)

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (feature_maps*2, 8, 8)

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (feature_maps*4, 4, 4)

            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # output: (1, 1, 1) -> probability the image is real
        )

    def forward(self, img):
        return self.net(img).view(-1)


def weights_init(m):
    """DCGAN paper's recommended weight init - matters a lot for GAN
    training stability."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
