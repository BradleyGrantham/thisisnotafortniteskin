import random

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt

from thisisnotafortniteskin import constants
from thisisnotafortniteskin.networks import discriminator, generator

random.seed(64)
np.random.seed(64)
torch.manual_seed(64)

REAL_LABEL = 1
FAKE_LABEL = 0


data_path = "../data/augmented"
BATCH_SIZE = 64
IMAGE_SIZE = 64

z = 100

# feature map size for generator
ngf = 64

# feature map size for generator
ndf = 64

# number of epochs
NUM_EPOCHS = 1000

# learning rate
LR = 0.0002

# hyperparam for optimisers
BETA1 = 0.5

if __name__ == "__main__":
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(BATCH_SIZE, z, 1, 1, device=constants.TORCH_DEVICE)

    discriminator_optimizer = optim.Adam(discriminator.parameters(),
                                         lr=LR, betas=(BETA1, 0.999))
    generator_optimizer = optim.Adam(generator.parameters(),
                                     lr=LR, betas=(BETA1, 0.999))

    img_list = []
    generator_losses = []
    discriminator_losses = []
    iters = 0

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(dataloader):

            discriminator.zero_grad()

            real_cpu = data[0].to(constants.TORCH_DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), REAL_LABEL, device=constants.TORCH_DEVICE)

            output = discriminator(real_cpu).view(-1)

            error_discriminator_real = criterion(output, label)

            error_discriminator_real.backward()
            discriminator_x = output.mean().item()

            # fake batch
            noise = torch.randn(b_size, z, 1, 1, device=constants.TORCH_DEVICE)

            fake = generator(noise)
            label.fill_(FAKE_LABEL)

            output = discriminator(fake.detach()).view(-1)

            error_discriminator_fake = criterion(output, label)

            error_discriminator_fake.backward()
            discriminator_generated_z1 = output.mean().item()

            error_discriminator = error_discriminator_real + error_discriminator_fake

            discriminator_optimizer.step()

            # generator optimisation
            generator.zero_grad()
            label.fill_(REAL_LABEL)

            output = discriminator(fake).view(-1)

            error_generator = criterion(output, label)

            error_generator.backward()
            discriminator_generated_z2 = output.mean().item()

            generator_optimizer.step()

            if i % 50 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, NUM_EPOCHS, i, len(dataloader),
                       error_discriminator.item(), error_generator.item(), discriminator_x, discriminator_generated_z1, discriminator_generated_z2))

            generator_losses.append(error_generator.item())
            discriminator_losses.append(error_discriminator.item())
            iters += 1

        # evaluate on some fixed noise and save the output
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
            plt.imshow(transforms.ToPILImage()(fake[0, :, :, :]))
            plt.savefig(str(epoch) + ".png")
        img_list.append(
            torchvision.utils.make_grid(fake, padding=2, normalize=True))

        torch.save(generator.state_dict(), "generator.pt")
