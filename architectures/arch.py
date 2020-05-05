from .holonet import HoloNet
from .discriminators import (Discriminator)

def get_network(z_dim, ngf, ndf):
    gen = HoloNet(z_dim=z_dim,
                  nf=ngf)
    #gen = Generator(z_dim=z_dim, nf=ngf)
    disc = Discriminator(nf=ndf,
                         sigmoid=True,
                         spec_norm=True,
                         n_classes=z_dim)
    return gen, disc
