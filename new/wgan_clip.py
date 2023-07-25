##########################################################################################################################################################
# Description: This is a PyTorch implementation of original WGAN (https://arxiv.org/abs/1701.0787) for MNIST, CIFAR10 and custom datasets taken from PyTorch tutorials and will be format for future models.
# The model is sensitive to hyperparameters and may not converge for most values.

# Here MNIST and CIFAR10 datasets are rescaled to 64x64x1 and 64*64x3 respectively thus increasing the number of parameters in the model with no real benefit.
# The model is trained for 20 epochs and the results are saved in the output folder.

# The dataset needs to defined as "python wgan_clip.py --dataset <dataset_name>" where dataset_name is either "mnist", "cifar10" or "custom" for custom datasets.
##########################################################################################################################################################
# Imports
import time
start = time.time()

import argparse
import os
from tqdm import tqdm, trange
import random

import torch

import torch.optim as optim
import torch.utils.data

import torchvision.utils as vutils

from utils import get_device, get_loader, save_plot_gan, weights_init
from model import Generator, Critic
##########################################################################################################################################################
# Arguments

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist | custom')
parser.add_argument('--dataroot', required=False, help='path to custom dataset')

parser.add_argument('--batchSize', type=int, default=64, help='input batch size')

parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', type=int, default=3, help='number of input channels')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--embedding_dim', type=int, default=20, help='size of the embedding vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ncf', type=int, default=64)

parser.add_argument('--conditional', action='store_true', help='use conditional GAN')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes for conditional GAN')

parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, default=0.00005')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, default=5')
parser.add_argument('--weight_clip', type=float, default=0.01, help='weight cliping, default=0.01')

parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--dry-run', action='store_true', help='check a single epoch works')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')

parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netC', default='', help="path to netC (to continue training)")
parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')

args = parser.parse_args()
print(args)

MODEL_NAME, _ = os.path.splitext(os.path.basename(os.path.realpath(__file__)))

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
if args.conditional:
    output_dir = str(args.outf + '/' + MODEL_NAME + '/' + args.dataset + '_conditional')
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

NUM_CLASSES = int(args.num_classes)

if str(args.dataset).lower() == 'mnist':
    CHANNELS_DIM = 1
    NUM_CLASSES = 10

elif str(args.dataset).lower() == 'cifar10':
    CHANNELS_DIM = 3
    NUM_CLASSES = 10

Z_DIM = int(args.nz)
EMBEDDING_DIM = int(args.embedding_dim)
NGF = int(args.ngf)
NCF = int(args.ncf)

EPOCHS = int(args.niter)
LEARNING_RATE = args.lr

CRITIC_ITER = int(args.critic_iter)
WEIGHT_CLIP = args.weight_clip

##########################################################################################################################################################
# Data
DATALOADER = get_loader(args, IMAGE_DIM, CHANNELS_DIM, BATCH_SIZE)

##########################################################################################################################################################
# Networks

netG = Generator(NGF, Z_DIM, CHANNELS_DIM, args.conditional, NUM_CLASSES, IMAGE_DIM, EMBEDDING_DIM).to(DEVICE)
netG.apply(weights_init)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
print(netG)

netC = Critic(NCF, CHANNELS_DIM, args.conditional, NUM_CLASSES, IMAGE_DIM).to(DEVICE)
netC.apply(weights_init)
if args.netC != '':
    netC.load_state_dict(torch.load(args.netC))
print(netC)

##########################################################################################################################################################
# Loss function and optimizer

#criterion = nn.BCELoss()

fixed_noise = torch.randn(args.batchSize, Z_DIM, 1, 1, device=DEVICE)
if args.conditional:
    cond_noise = torch.randn(NUM_CLASSES*4, Z_DIM, 1, 1, device=DEVICE)
    cond_labels = torch.range(0, NUM_CLASSES-1, dtype=torch.long).repeat(4).to(DEVICE)

optimizerC = optim.RMSprop(netC.parameters(), lr=LEARNING_RATE)
optimizerG = optim.RMSprop(netG.parameters(), lr=LEARNING_RATE)

##########################################################################################################################################################
# Training Loop
if args.dry_run:
    args.niter = 2

losses_g = []
losses_c = []

for epoch in trange(args.niter, unit='epoch', desc='Training'):

    loss_c = 0.0
    loss_g = 0.0

    with tqdm(DATALOADER, desc="Train") as tbatch:

        for i, data in enumerate(tbatch, 0):
            
            loss_c_batch = 0.0
            loss_g_batch = 0.0

            ############################
            # (1) Update D network: miniimize E(-C(x)) + E(C(G(z))
            ###########################

            for _ in range(CRITIC_ITER):
                # train with real
                netC.zero_grad()
                errC_real=0.0

                real, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                batch_size = real.size(0)

                if args.conditional:

                    output = netC(real, labels)
                    errC_real = -output.mean()

                    noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
                    fake = netG(noise, labels)

                    output = netC(fake.detach(), labels)
                    errC_fake = output.mean()

                    errC = errC_real + errC_fake
                    errC.backward(retain_graph=True)

                    optimizerC.step()

                    loss_c_batch += errC.item()

                    for para in netC.parameters():
                        para.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

                else:
                    output = netC(real)
                    errC_real = -output.mean()

                    noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
                    fake = netG(noise)

                    output = netC(fake.detach())
                    errC_fake = output.mean()

                    errC = errC_real + errC_fake
                    errC.backward(retain_graph=True)

                    optimizerC.step()

                    loss_c_batch += errC.item()

                    for para in netC.parameters():
                        para.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            loss_c_batch/=CRITIC_ITER

            ############################
            # (2) Update G network: minimize E(-C(G(z)))
            ###########################

            if args.conditional:
                netG.zero_grad()

                output = netC(fake, labels)
                errG = -output.mean()
                errG.backward(retain_graph=True)

                optimizerG.step()

                loss_g_batch += errG.item()

            else:
                netG.zero_grad()

                output = netC(fake)
                errG = -output.mean()
                errG.backward(retain_graph=True)

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
                if args.conditional:
                    vutils.save_image(real.detach(),
                        '%s/real_samples_epoch_%03d.png' % (output_dir, epoch),
                        normalize=True)
                    fake = netG(fixed_noise, labels)
                    vutils.save_image(fake.detach(),
                            '%s/fake_samples_epoch_%03d.png' % (output_dir, epoch),
                            normalize=True)
                    cond_fake = netG(cond_noise, cond_labels)
                    vutils.save_image(cond_fake.detach(),
                            '%s/cond_fake_samples_epoch_%03d.png' % (output_dir, epoch),
                            normalize=True)
                else:
                    vutils.save_image(real.detach(),
                            '%s/real_samples.png' % (output_dir),
                            normalize=True)
                    fake = netG(fixed_noise)
                    vutils.save_image(fake.detach(),
                            '%s/fake_samples_epoch_%03d.png' % (output_dir, epoch),
                            normalize=True)
                #vutils.save_image(interpolated_images.detach(),
                #       '%s/interpolated_samples_epoch_%03d.png' % (str(args.outf + '/' + args.dataset), epoch),
                #       normalize=True)
                
            tbatch.set_postfix(loss_C=loss_c_batch, loss_G=loss_g_batch)

            loss_c += loss_c_batch
            loss_g += loss_g_batch

            if args.dry_run:
                break

        losses_c.append(loss_c)

        losses_g.append(loss_g)

        if epoch % 5 == 0:
            torch.save(netG.state_dict(), '%s/netG_epoch_%03d.pth' % (output_dir,epoch))
            torch.save(netC.state_dict(), '%s/netC_epoch_%03d.pth' % (output_dir,epoch))

##########################################################################################################################################################
# Plot losses

save_plot_gan(losses_g, losses_c, output_dir, MODEL_NAME)

##########################################################################################################################################################

print('It took', time.time()-start, 'seconds.')