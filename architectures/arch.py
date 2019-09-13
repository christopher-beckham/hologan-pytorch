from .holonet import HoloNet
from .discriminators import (Discriminator,
                             MI)

def get_network(z_dim, ngf, ndf, use_64px=False, z_extra_fc=False):
    gen = HoloNet(z_dim=z_dim,
                  nf=ngf,
                  use_64px=use_64px)
    #gen = Generator(z_dim=z_dim, nf=ngf)
    disc = Discriminator(nf=ndf,
                         z_dim=z_dim,
                         sigmoid=True,
                         spec_norm=True,
                         lite=False if use_64px else True,
                         z_extra_fc=z_extra_fc)
    #mi = MI(nf=nmf,
    #        z_dim=z_dim)
    return gen, disc #, mi
