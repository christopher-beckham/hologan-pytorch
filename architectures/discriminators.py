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
                 input_nc=3,
                 n_out=1,
                 n_classes=0,
                 sigmoid=False,
                 spec_norm=False):
        """
        """
        super(Discriminator, self).__init__()

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x : x

        self.relu = nn.ReLU()

        self.base = nn.Sequential(
            FirstResBlockDiscriminator(input_nc, nf,
                                       stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf, nf*2,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*2, nf*4,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*4, nf*8,
                                  stride=2, spec_norm=spec_norm),
        )

        self.d = ResBlockDiscriminator(nf*8, nf*8,
                                       spec_norm=spec_norm)
        self.q = ResBlockDiscriminator(nf*8, nf*8,
                                       spec_norm=spec_norm)

        self.fc = nn.Linear(nf*8, n_out)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = self.spec_norm(self.fc)

        self.pool = nn.AvgPool2d(4)

        if n_classes > 0:
            self.cls = nn.Linear(nf*8, n_classes+3)
            nn.init.xavier_uniform(self.cls.weight.data, 1.)
            self.cls = self.spec_norm(self.cls)
        else:
            self.cls = None

        self.sigmoid = sigmoid
        self.n_classes = n_classes

    #def encode(self, x):
    #    x = self.base(x)
    #    return x

    def forward(self, x):
        h = self.base(x)

        h_d = self.pool(self.d(h))
        h_q = self.pool(self.q(h))

        h_d = h_d.view(-1, h_d.size(1))
        h_q = h_q.view(-1, h_q.size(1))

        pred_d = self.fc(h_d)
        if self.sigmoid:
            pred_d = F.sigmoid(pred_d)

        if self.cls is not None:
            pred_zt = self.cls(h_q)
            pred_z = pred_zt[:, 0:self.n_classes]
            pred_t = pred_zt[:, self.n_classes:]
        else:
            pred_z, pred_t = None
        return pred_d, pred_z, pred_t
