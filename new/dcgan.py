##########################################################################################################################################################
# Description: This is a PyTorch implementation of DCGAN (https://arxiv.org/abs/1511.06434) for MNIST, CIFAR10 and custom datasets taken from PyTorch tutorials and will be format for future models.
# The model is sensitive to hyperparameters and may not converge for most values.

# Here MNIST and CIFAR10 datasets are rescaled to 64x64x1 and 64*64x3 respectively thus increasing the number of parameters in the model with no real benefit.
# The model is trained for 20 epochs and the results are saved in the output folder.

# The dataset needs to defined as "python dcgan.py --dataset <dataset_name>" where dataset_name is either "mnist", "cifar10" or "custom" for custom datasets.
##########################################################################################################################################################
# Imports
import time
start = time.time()

import argparse
import os
from tqdm import tqdm, trange
import random

import torch
import torch.nn as nn

import torch.optim as optim
import torch.utils.data

import torchvision.utils as vutils

from utils import *
from model import Generator, Discriminator

##########################################################################################################################################################
# Arguments

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist | custom')
parser.add_argument('--dataroot', required=False, help='path to custom dataset')

parser.add_argument('--batchSize', type=int, default=64, help='input batch size')

parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', type=int, default=3, help='number of input channels')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)

parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')

parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--dry-run', action='store_true', help='check a single epoch works')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')

parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')

args = parser.parse_args()
print(args)

MODEL_NAME = 'dcgan'

##########################################################################################################################################################
# Set random seed for reproducibility

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

##########################################################################################################################################################
# Create output directory
output_dir = str(args.outf + '/' + MODEL_NAME + '/' + args.dataset)
try:
    os.makedirs(output_dir)
except OSError:
    pass

##########################################################################################################################################################
# Device configuration

DEVICE = get_device(args)

##########################################################################################################################################################
# Parameters

BATCH_SIZE = int(args.batchSize)

IMAGE_DIM = args.imageSize
CHANNELS_DIM = args.channels

if str(args.dataset).lower() == 'mnist':
    CHANNELS_DIM = 1
elif str(args.dataset).lower() == 'cifar10':
    CHANNELS_DIM = 3

Z_DIM = int(args.nz)
NGF = int(args.ngf)
NDF = int(args.ndf)

EPOCHS = int(args.niter)
LEARNING_RATE = args.lr
BETA1 = args.beta1
BETA2 = args.beta2

##########################################################################################################################################################
# Data
DATALOADER = get_loader(args, IMAGE_DIM, CHANNELS_DIM, BATCH_SIZE)

##########################################################################################################################################################
# Networks

netG = Generator(NGF, Z_DIM, CHANNELS_DIM).to(DEVICE)
netG.apply(weights_init)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
print(netG)

netD = Discriminator(NDF, CHANNELS_DIM).to(DEVICE)
netD.apply(weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)

##########################################################################################################################################################
# Loss function and optimizer

criterion = nn.BCELoss()

fixed_noise = torch.randn(args.batchSize, Z_DIM, 1, 1, device=DEVICE)

optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

##########################################################################################################################################################
# Training Loop
if args.dry_run:
    args.niter = 2

losses_g = []
losses_d = []

for epoch in trange(args.niter, unit='epoch', desc='Training'):

    loss_d = 0.0
    loss_g = 0.0

    with tqdm(DATALOADER, desc="Train") as tbatch:

        for i, data in enumerate(tbatch, 0):
            
            loss_d_batch = 0.0
            loss_g_batch = 0.0

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # train with real
            netD.zero_grad()
            real = data[0].to(DEVICE)
            batch_size = real.size(0)

            output = netD(real)
            errD_real = criterion(output, torch.ones_like(output))
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
            fake = netG(noise)

            output = netD(fake.detach())
            errD_fake = criterion(output, torch.zeros_like(output))
            D_G_z1 = output.mean().item()

            errD = (errD_real + errD_fake)/2
            errD.backward(retain_graph=True)
            optimizerD.step()

            loss_d_batch += errD.item()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            netG.zero_grad()

            output = netD(fake)
            errG = criterion(output, torch.ones_like(output))

            errG.backward(retain_graph=True)
            D_G_z2 = output.mean().item()
            optimizerG.step()

            loss_g_batch += errG.item()


            # Interpolate between two points in latent space
            #point_1 = torch.randn((1, Z_DIM, 1, 1)).to(DEVICE)
            #point_2 = torch.randn((1, Z_DIM, 1, 1)).to(DEVICE)
            #interpolation = point_1.detach().clone()
            #for i in range(1, 16, 1):
            #    inter = torch.lerp(point_1, point_2,(i/15.0)).to(DEVICE)
            #    interpolation = torch.cat((interpolation, inter), 0).to(DEVICE)
            #interpolated_images = netG(interpolation)

            if i % 100 == 0:
                vutils.save_image(real.detach(),
                        '%s/real_samples.png' % output_dir,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (output_dir, epoch),
                        normalize=True)
                #vutils.save_image(interpolated_images.detach(),
                #       '%s/interpolated_samples_epoch_%03d.png' % (str(args.outf + '/' + args.dataset), epoch),
                #       normalize=True)
                
            tbatch.set_postfix(loss_D=errD.item(), loss_G=errG.item(), D_x=D_x, D_G_z1=D_G_z1, D_G_z2=D_G_z2)

            loss_d += loss_d_batch
            loss_g += loss_g_batch

            if args.dry_run:
                break

        losses_d.append(loss_d)
        losses_g.append(loss_g)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG.pth' % (output_dir))
        torch.save(netD.state_dict(), '%s/netD.pth' % (output_dir))

##########################################################################################################################################################
# Plot losses

save_plot_gan(losses_g, losses_d, output_dir, MODEL_NAME)

##########################################################################################################################################################

print('It took', time.time()-start, 'seconds.')