'''
MIT License

Copyright (c) 2019 Christopher Beckham
Copyright (c) 2017 Christian Cosgrove

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import numpy as np

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.spec_norm(self.conv1),
                nn.ReLU(),
                self.spec_norm(self.conv2)
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.spec_norm(self.conv1),
                nn.ReLU(),
                self.spec_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        self.bypass = nn.Sequential()
        if in_channels != out_channels:
            self.bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass.weight.data, np.sqrt(2))
            self.bypass = self.spec_norm(self.bypass)
        if stride != 1:
            self.bypass = nn.Sequential(
                self.bypass,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.spec_norm(self.conv1),
            nn.ReLU(),
            self.spec_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.spec_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class Discriminator(nn.Module):
    def __init__(self,
                 nf,
                 z_dim,
                 input_nc=3,
                 n_out=1,
                 sigmoid=False,
                 z_extra_fc=False,
                 lite=False,
                 spec_norm=False):
        super(Discriminator, self).__init__()

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x : x

        self.model = nn.Sequential(
            FirstResBlockDiscriminator(input_nc, nf,
                                       stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf, nf*2,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*2, nf*4,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*4, nf*8,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*8, nf*8,
                                  spec_norm=spec_norm) if not lite else nn.Identity(),
            nn.ReLU(),
            #nn.AvgPool2d(4),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(nf*8, n_out)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = self.spec_norm(self.fc)

        if not z_extra_fc:
            self.fc_z = nn.Linear(nf*8, z_dim)
            nn.init.xavier_uniform(self.fc_z.weight.data, 1.)
            self.fc_z = self.spec_norm(self.fc_z)
        else:
            fc1 = nn.Linear(nf*8, nf*4)
            nn.init.xavier_uniform(fc1.weight.data, 1.)
            fc2 = nn.Linear(nf*4, z_dim)
            nn.init.xavier_uniform(fc2.weight.data, 1.)
            self.fc_z = nn.Sequential(
                self.spec_norm(fc1),
                nn.ReLU(),
                self.spec_norm(fc2)
            )

        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.size(1))
        result = self.fc(x)
        pred_z = self.fc_z(x)
        if self.sigmoid:
            result = F.sigmoid(result)
        return result, pred_z

class MI(nn.Module):
    def __init__(self,
                 nf,
                 z_dim,
                 input_nc=3):
        super(MI, self).__init__()

        model = [nn.Conv2d(input_nc, nf, kernel_size=3, padding=1),
                 nn.InstanceNorm2d(nf),
                 nn.ReLU()]
        n_ds = 3
        for i in range(n_ds):
            mult = 2**i
            mult2 = 2**(i+1)
            model += [
                nn.Conv2d(nf*mult, nf*mult2, kernel_size=3, padding=1, stride=2),
                nn.InstanceNorm2d(nf*mult2),
                nn.ReLU()
            ]
        model += [nn.AdaptiveAvgPool2d(1)]
        self.model = nn.Sequential(*model)

        self.fc_z = nn.Linear(nf*mult2, z_dim)
        nn.init.xavier_uniform(self.fc_z.weight.data, 1.)

        self.fc_theta = nn.Linear(nf*mult2, 3)
        nn.init.xavier_uniform(self.fc_theta.weight.data, 1.)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.size(1))
        pred_z = self.fc_z(x)
        pred_theta = self.fc_theta(x)
        return pred_z, pred_theta
