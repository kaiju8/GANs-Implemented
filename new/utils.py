import torch
from torch import nn

import torchvision.datasets as dset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


def get_device(args):
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    elif torch.backends.mps.is_available() and not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps")

    use_mps = args.mps and torch.backends.mps.is_available()
    use_cuda = args.cuda and torch.cuda.is_available()

    if use_cuda:
        return torch.device("cuda")
    elif use_mps:
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def get_loader(args, IMAGE_DIM, CHANNELS_DIM, BATCH_SIZE):
    if args.dataroot is None and str(args.dataset).lower() == 'custom':
        raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % args.dataset)

    if args.dataset in ['custom']:
        # custom dataset
        dataset = dset.ImageFolder(root=args.dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(IMAGE_DIM),
                                    transforms.CenterCrop(IMAGE_DIM),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5 for _ in range(CHANNELS_DIM)], [0.5 for _ in range(CHANNELS_DIM)]),
                                ]))
        
    elif args.dataset == 'cifar10':
        dataset = dset.CIFAR10(root='./data', download=True,
                            transform=transforms.Compose([
                                transforms.Resize(IMAGE_DIM),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5 for _ in range(CHANNELS_DIM)], [0.5 for _ in range(CHANNELS_DIM)]),
                            ]))

    elif args.dataset == 'mnist':
        dataset = dset.MNIST(root='./data', download=True,
                        transform=transforms.Compose([
                            transforms.Resize(IMAGE_DIM),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5 for _ in range(CHANNELS_DIM)], [0.5 for _ in range(CHANNELS_DIM)]),
                            ]))

    assert dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return loader

def save_plot_gan(losses_g, losses_o, output_dir, model_name):

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss")
    plt.plot(losses_g, label='Generator loss')
    if 'dcgan' in model_name:
        plt.plot(losses_o, label='Discriminator Loss')
    elif 'wgan' in model_name:
        plt.plot(losses_o, label='Critic Loss')
    plt.legend()
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig('%s/loss.png' % output_dir)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def gradient_penalty(interpolated_score, interpolated_image, device):
    gradient = torch.autograd.grad(outputs=interpolated_score, inputs=interpolated_image,
                                    grad_outputs=torch.ones(interpolated_score.size()).to(device),
                                    create_graph=True, retain_graph=True)[0]
    gradient = gradient.view(gradient.size(0), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty