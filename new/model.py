import torch
from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ngf , z_dim, channels_dim, conditional, num_classes, image_dim, embed_length, batch_norm=True):
        super(Generator, self).__init__()
        if conditional:
            z_dim += embed_length
        self.gen = nn.Sequential(
            # input is Z, going into a convolution
            self.block(z_dim, ngf * 8, kernel_size=4, stride=1, padding=0, batch_norm=batch_norm),
            # state size. (ngf *8) x 4 x 4
            self.block(ngf * 8, ngf * 4, batch_norm=batch_norm),
            # state size. (ngf *4) x 8 x 8
            self.block(ngf * 4, ngf * 2, batch_norm=batch_norm),
            # state size. (ngf *2) x 16 x 16
            self.block(ngf * 2, ngf , batch_norm=batch_norm),
            # state size. (ngf ) x 32 x 32
            nn.ConvTranspose2d( ngf ,channels_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (channels_dim) x 64 x 64
        )
        if conditional:
            self.embed = nn.Embedding(num_classes, embed_length)
        

    def forward(self, input, labels=None):
        if labels is not None:
            labels = self.embed(labels).unsqueeze(2).unsqueeze(3)
            input = torch.cat((input, labels), dim=1)
        output = self.gen(input)
        return output
    
    def block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, relu=True):
        layers = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if relu:
            layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)
    
class Discriminator(nn.Module):
    def __init__(self, ndf, channels_dim, conditional, num_classes, image_dim, batch_norm=True):
        super(Discriminator, self).__init__()
        if conditional:
            channels_dim += 1
        self.disc = nn.Sequential(
            # input is (channels_dim) x 64 x 64
            self.block(channels_dim, ndf, batch_norm=False),
            # state size. (ndf) x 32 x 32
            self.block(ndf, ndf * 2, batch_norm=batch_norm),
            # state size. (ndf*2) x 16 x 16
            self.block(ndf * 2, ndf * 4, batch_norm=batch_norm),
            # state size. (ndf*4) x 8 x 8
            self.block(ndf * 4, ndf * 8, batch_norm=batch_norm),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        if conditional:
            self.embed = nn.Embedding(num_classes, image_dim*image_dim)

    def forward(self, input, labels=None):
        if labels is not None:
            labels = self.embed(labels).view(labels.size(0), 1, input.size(2), input.size(3))
            input = torch.cat((input, labels), dim=1)
        output = self.critic(input)
        return output.view(-1, 1).squeeze(1)
    
    def block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, relu=True):
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if relu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
class Critic(nn.Module):
    def __init__(self, ncf , channels_dim, conditional, num_classes, image_dim, instance_norm=False, relu=True):
        super(Critic, self).__init__()
        if conditional:
            channels_dim += 1
        self.critic = nn.Sequential(
            # input is (channels_dim) x 64 x 64
            nn.Conv2d(channels_dim, ncf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ncf ) x 32 x 32
            self.block(ncf , ncf * 2, instance_norm=instance_norm, relu=relu),
            # state size. (ncf *2) x 16 x 16
            self.block(ncf * 2, ncf * 4, instance_norm=instance_norm, relu=relu),
            # state size. (ncf *4) x 8 x 8
            self.block(ncf * 4, ncf * 8, instance_norm=instance_norm, relu=relu),
            # state size. (ncf *8) x 4 x 4
            nn.Conv2d(ncf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # No Sigmod here
        )
        if conditional:
            self.embed = nn.Embedding(num_classes, image_dim*image_dim)


    def forward(self, input, labels=None):
        if labels is not None:
            labels = self.embed(labels).view(labels.size(0), 1, input.size(2), input.size(3))
            input = torch.cat((input, labels), dim=1)
        output = self.critic(input)
        return output.view(-1, 1).squeeze(1)
    
    def block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, instance_norm=False, relu=True):
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        if instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        else:
            layers.append(nn.BatchNorm2d(out_channels))
        if relu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
class VAE(nn.Module):
    def __init__(self, input_dim, z_dim=100, h_dim=2000):
        super().__init__()
        self.img2hid = nn.Linear(input_dim, h_dim)
        self.hid2mu = nn.Linear(h_dim, z_dim)
        self.hid2sigma = nn.Linear(h_dim, z_dim)
        
        self.z2hid = nn.Linear(z_dim, h_dim)
        self.hid2img = nn.Linear(h_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.img2hid(x))
        mu, sigma = self.hid2mu(h), self.hid2sigma(h)
        
        return mu, sigma
    
    def reparametrize(self, mu, sigma):
        std = torch.exp(sigma/2)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    
    def decode(self, z):
        h_ = self.z2hid(z)
        x_ = torch.sigmoid(self.hid2img(h_))
        
        return x_
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        
        z_ = self.reparametrize(mu, sigma)
        x_ = self.decode(z_)
        
        return x_, mu, sigma
