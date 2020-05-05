import torch
import numpy as np
from torchvision.utils import save_image

def count_params(module, trainable_only=True):
    """Count the number of parameters in a
    module.
    :param module: PyTorch module
    :param trainable_only: only count trainable
      parameters.
    :returns: number of parameters
    :rtype:
    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num

def generate_rotations(gan,
                       z_batch,
                       out_folder,
                       axis='x',
                       min_angle=None,
                       max_angle=None,
                       num=5):
    if min_angle is None:
        min_angle = gan.angles['min_angle_%s' % axis]
    if max_angle is None:
        max_angle = gan.angles['max_angle_%s' % axis]
    from itertools import chain
    linspace = chain(np.linspace(min_angle, max_angle, num),
                     np.linspace(max_angle, min_angle, num))
    with torch.no_grad():
        for idx,p in enumerate(linspace):
            #enc_rot = gan.rotate_random(enc, angle=p)
            angles = np.zeros((z_batch.size(0), 3)).astype(np.float32)
            angles[:, gan.rot2idx[axis]] += p
            thetas = gan.get_theta(angles)
            if z_batch.is_cuda:
                thetas = thetas.cuda()
            x_fake = gan.g(z_batch, thetas)
            #pbuf.append(x_fake*0.5 + 0.5)
            save_image(x_fake*0.5 + 0.5,
                       "%s/{0:06d}.png".format(idx) % (out_folder))
