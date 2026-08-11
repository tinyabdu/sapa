"""
image_gen/train_gan.py

Trains the DCGAN on Fashion-MNIST (swap in any small image dataset -
see NOTES at the bottom for using a Kaggle dataset instead).

DESIGNED FOR GOOGLE COLAB:
1. Upload this file + gan_model.py to Colab (or clone from GitHub)
2. Runtime -> Change runtime type -> GPU (T4, free tier)
3. Run: !python train_gan.py
4. Checkpoints save to checkpoints/, sample images save to samples/
5. Download checkpoints/generator_final.pth when done - that's what
   you deploy on Railway for inference.

Training Fashion-MNIST to decent quality takes ~30-50 epochs, roughly
20-40 minutes on a free Colab T4 GPU.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from gan_model import Generator, Discriminator, weights_init

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
LATENT_DIM = 100
IMG_CHANNELS = 1          # 1 for grayscale (Fashion-MNIST/MNIST), 3 for RGB
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.0002
BETA1 = 0.5                # Adam beta1, DCGAN paper recommendation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("samples", exist_ok=True)

print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------
# Data - Fashion-MNIST, resized to 32x32 to match the GAN architecture
# ---------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # scale to [-1, 1] to match Tanh output
])

dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
netG = Generator(LATENT_DIM, IMG_CHANNELS).to(DEVICE)
netD = Discriminator(IMG_CHANNELS).to(DEVICE)
netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)  # for consistent progress snapshots

REAL_LABEL = 1.0
FAKE_LABEL = 0.0

# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
for epoch in range(EPOCHS):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(DEVICE)
        b_size = real_images.size(0)

        # --- Train Discriminator: maximize log(D(real)) + log(1 - D(G(z))) ---
        netD.zero_grad()
        label = torch.full((b_size,), REAL_LABEL, device=DEVICE)
        output = netD(real_images)
        loss_d_real = criterion(output, label)
        loss_d_real.backward()

        noise = torch.randn(b_size, LATENT_DIM, device=DEVICE)
        fake_images = netG(noise)
        label.fill_(FAKE_LABEL)
        output = netD(fake_images.detach())
        loss_d_fake = criterion(output, label)
        loss_d_fake.backward()

        loss_d = loss_d_real + loss_d_fake
        optimizerD.step()

        # --- Train Generator: maximize log(D(G(z))) ---
        netG.zero_grad()
        label.fill_(REAL_LABEL)  # generator wants discriminator to say "real"
        output = netD(fake_images)
        loss_g = criterion(output, label)
        loss_g.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{i}/{len(dataloader)}] "
                  f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}")

    # Save a sample grid of generated images after each epoch, so you
    # can watch quality improve over training
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    vutils.save_image(fake, f"samples/epoch_{epoch+1:03d}.png", normalize=True, nrow=8)

    # Checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(netG.state_dict(), f"checkpoints/generator_epoch_{epoch+1}.pth")

# Final save - this is the file you download and deploy
torch.save(netG.state_dict(), "checkpoints/generator_final.pth")
print("\nTraining complete. Deploy checkpoints/generator_final.pth to Railway.")

# ---------------------------------------------------------------------
# NOTES: using a Kaggle dataset instead of Fashion-MNIST
# ---------------------------------------------------------------------
# 1. Download a Kaggle image dataset (folder of images, one class or many)
# 2. Replace the `dataset = torchvision.datasets.FashionMNIST(...)` block with:
#
#    dataset = torchvision.datasets.ImageFolder(
#        root="path/to/your/kaggle/images",
#        transform=transform
#    )
#
# 3. If images are RGB (color), set IMG_CHANNELS = 3 above, and update
#    transform's Normalize to `transforms.Normalize([0.5]*3, [0.5]*3)`
# 4. Larger/more complex images (faces, objects) need more epochs and
#    benefit from a bigger feature_maps value in gan_model.py (try 128)
