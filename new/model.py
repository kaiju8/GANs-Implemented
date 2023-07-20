import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngf , z_dim, channels_dim, batch_norm=True):
        super(Generator, self).__init__()

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

    def forward(self, input):
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
    def __init__(self, ndf, channels_dim, batch_norm=True):
        super(Discriminator, self).__init__()

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

    def forward(self, input):

        output = self.disc(input)
        return output.view(-1, 1).squeeze(1)
    
    def block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, relu=True):
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if relu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
class Critic(nn.Module):
    def __init__(self, ncf , channels_dim, instance_norm=False, relu=True):
        super(Critic, self).__init__()

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

    def forward(self, input):
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
