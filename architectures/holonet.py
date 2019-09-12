import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

class ResBlock2d(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ResBlock2d, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1)
        self.bn = nn.InstanceNorm2d(in_ch)
        self.relu = nn.LeakyReLU()
        self.bn2 = nn.InstanceNorm2d(out_ch)

        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        bypass = []
        if in_ch != out_ch:
            bypass.append(nn.Conv2d(in_ch, out_ch, 1, 1))
        self.bypass = nn.Sequential(*bypass)

    def forward(self, inp):
        x = self.bn(inp)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + self.bypass(inp)

class ResBlock3d(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ResBlock3d, self).__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, 1, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, padding=1)
        self.bn = nn.InstanceNorm3d(in_ch)
        self.relu = nn.LeakyReLU()
        self.bn2 = nn.InstanceNorm3d(out_ch)

        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        bypass = []
        if in_ch != out_ch:
            bypass.append(nn.Conv3d(in_ch, out_ch, 1, 1))
        self.bypass = nn.Sequential(*bypass)

    def forward(self, inp):
        x = self.bn(inp)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + self.bypass(inp)



def _adain_module_3d(z_dim, out_ch):
    adain = nn.InstanceNorm3d(out_ch, affine=True)
    z_mlp = nn.Sequential(
        nn.Linear(z_dim, out_ch*2), # both var and mean
    )
    return adain, z_mlp

def _adain_module_2d(z_dim, out_ch):
    adain = nn.InstanceNorm2d(out_ch, affine=True)
    z_mlp = nn.Linear(z_dim, out_ch*2)
    return adain, z_mlp

class HoloNet(nn.Module):

    def __init__(self, nf, out_ch=3, z_dim=128, use_64px=False):
        super(HoloNet, self).__init__()

        self.ups_3d = nn.Upsample(scale_factor=2, mode='nearest')
        self.ups_2d = nn.Upsample(scale_factor=2, mode='nearest')

        self.z_dim = z_dim

        xstart = ( torch.randn((1, nf, 4, 4, 4)) - 0.5 ) / 0.5
        nn.init.xavier_uniform(xstart.data, 1.)
        self.xstart = nn.Parameter(xstart)
        self.xstart.requires_grad = True

        self.nf = nf

        self.rb1 = ResBlock3d(nf, nf // 2)
        self.adain_1, self.z_mlp1 = _adain_module_3d(z_dim, nf//2)

        self.rb2 = ResBlock3d(nf // 2, nf // 4)
        self.adain_2, self.z_mlp2 = _adain_module_3d(z_dim, nf//4)

        # The 3d transformation is done here.

        # Two convs (no adain) that bring it from nf//4 to nf//8
        self.postproc = nn.Sequential(
            nn.Conv3d(nf//4, nf//8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(nf//8, affine=True),
            nn.ReLU(),
            nn.Conv3d(nf//8, nf//8, kernel_size=3, padding=1),
            nn.InstanceNorm3d(nf//8, affine=True),
            nn.ReLU()
        )

        # Then concatenation happens.

        #resblocks = [ResBlock3d(nf//4, nf//4) for _ in range(num_blocks)]
        #self.rbs = nn.Sequential(*resblocks)

        pnf = (nf//8)*(4**2) # 512

        # TODO: should be 1x1
        self.proj = nn.Sequential(
            nn.Conv2d(pnf, pnf//2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(pnf//2, affine=True),
            nn.ReLU()
        )

        self.rb1_2d = ResBlock2d(pnf//2, pnf//4)
        self.adain_3, self.z_mlp3 = _adain_module_2d(z_dim, pnf//4)

        if use_64px:
            self.rb2_2d = ResBlock2d(pnf//4, pnf//8)
            self.adain_4, self.z_mlp4 = _adain_module_2d(z_dim, pnf//8)
            self.conv_final = nn.Conv2d(pnf//8, out_ch, 3, padding=1)
        else:
            self.conv_final = nn.Conv2d(pnf//4, out_ch, 3, padding=1)

        #self.rb3_2d = ResBlock2d(nf//8, nf//16)
        #self.adain_5, self.z_mlp5 = _adain_module_2d(z_dim, nf//16)

        self.tanh = nn.Tanh()
        self.use_64px = use_64px

    def stn(self, x, theta):
        # theta must be (Bs, 3, 4) = [R|t]
        #theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        img = F.grid_sample(x, grid)

        return img

    def _rshp2d(self, z):
        return z.view(-1, z.size(1), 1, 1)

    def _rshp3d(self, z):
        return z.view(-1, z.size(1), 1, 1, 1)

    def _split(self, z):
        len_ = z.size(1)
        mean = z[:, 0:(len_//2)]
        var = F.softplus(z[:, (len_//2):])
        return mean, var

    def forward(self, z, thetas):

        bs = z.size(0)
        # (512, 4, 4, 4)
        xstart = self.xstart.repeat((bs, 1, 1, 1, 1))

        # (256, 8, 8, 8)
        h1 = self.adain_1(self.ups_3d(self.rb1(xstart)))
        z1_mean, z1_var = self._split(self._rshp3d(self.z_mlp1(z)))
        h1 = h1*z1_var + z1_mean

        # (128, 16, 16, 16)
        h2 = self.adain_2(self.ups_3d(self.rb2(h1)))
        z2_mean, z2_var = self._split(self._rshp3d(self.z_mlp2(z)))
        h2 = h2*z2_var + z2_mean

        # Perform rotation
        h2_rotated = self.stn(h2, thetas)

        #h4 = self.rbs(h2_rotated)
        # (64, 16, 16, 16)
        # (64, 16, 16, 16)
        h4 = self.postproc(h2_rotated)

        # Projection unit. Concat depth and channels
        # (32*16, 16, 16) = (512, 16, 16)
        h4_proj = h4.view(-1, h4.size(1)*h4.size(2), h4.size(3), h4.size(4))

        # (256, 16, 16) (TODO: this should be a 1x1 conv)
        h4_proj = self.proj(h4_proj)

        # Now upconv your way to the original spatial dim.

        # (128, 32, 32)
        h5 = self.adain_3(self.ups_2d(self.rb1_2d(h4_proj)))
        z3_mean, z3_var = self._split(self._rshp2d(self.z_mlp3(z)))
        h5 = h5*z3_var + z3_mean
        h_last = h5

        if self.use_64px:
            h6 = self.adain_4(self.ups_2d(self.rb2_2d(h5)))
            z4_mean, z4_var = self._split(self._rshp2d(self.z_mlp4(z)))
            h6 = h6*z4_var + z4_mean
            h_last = h6

        #h7 = self.adain_5(self.ups_2d(self.rb3_2d(h6)))
        #z5_mean, z5_var = self._split(self._rshp2d(self.z_mlp5(z)))
        #h7 = h7*z5_var + z5_mean

        # (3, 32, 32)
        h_final = self.tanh(self.conv_final(h_last))

        return h_final
