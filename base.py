import torch
import os
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch import optim
from torch.autograd import grad
from skimage.io import imsave
from torchvision.utils import save_image
from itertools import chain

class GAN:
    """
    Base model for GAN models
    """
    def __init__(self,
                 gen_fn,
                 disc_fn,
                 z_dim,
                 lamb=0.,
                 beta=0.,
                 opt_g=optim.Adam,
                 opt_d=optim.Adam,
                 opt_mi=optim.Adam,
                 opt_d_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 opt_g_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 opt_mi_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 update_g_every=5,
                 handlers=[],
                 scheduler_fn=None,
                 scheduler_args={},
                 use_cuda='detect'):
        assert use_cuda in [True, False, 'detect']
        if use_cuda == 'detect':
            use_cuda = True if torch.cuda.is_available() else False
        self.z_dim = z_dim
        self.update_g_every = update_g_every
        self.g = gen_fn
        self.d = disc_fn
        self.lamb = lamb
        self.beta = beta
        optim_g = opt_g(filter(lambda p: p.requires_grad,
                               self.g.parameters()), **opt_g_args)
        optim_d = opt_d(filter(lambda p: p.requires_grad,
                               self.d.parameters()), **opt_d_args)
        self.optim = {
            'g': optim_g,
            'd': optim_d,
        }
        # HACK: this is actually both the MI network
        # and G.
        self.optim_mi = opt_mi(filter(lambda p: p.requires_grad,
                                      chain(self.d.parameters(),
                                            self.g.parameters())), **opt_mi_args)

        self.scheduler = {}
        if scheduler_fn is not None:
            for key in self.optim:
                self.scheduler[key] = scheduler_fn(
                    self.optim[key], **scheduler_args)
        self.handlers = handlers
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.g.cuda()
            self.d.cuda()
        self.last_epoch = 0

    def _get_stats(self, dict_, mode):
        stats = OrderedDict({})
        for key in dict_.keys():
            stats[key] = np.mean(dict_[key])
        return stats

    def sample_z(self, bs, seed=None):
        """Return a sample z ~ p(z)"""
        if seed is not None:
            rnd_state = np.random.RandomState(seed)
            z = torch.from_numpy(
                rnd_state.normal(0, 1, size=(bs, self.z_dim))
            ).float()
        else:
            z = torch.from_numpy(
                np.random.normal(0, 1, size=(bs, self.z_dim))
            ).float()
        if self.use_cuda:
            z = z.cuda()
        return z

    def loss(self, prediction, target):
        if not hasattr(target, '__len__'):
            target = torch.ones_like(prediction)*target
            if prediction.is_cuda:
                target = target.cuda()
        loss = torch.nn.BCELoss()
        if prediction.is_cuda:
            loss = loss.cuda()
        return loss(prediction, target)

    def _train(self):
        self.g.train()
        self.d.train()

    def _eval(self):
        self.g.eval()
        self.d.eval()

    def train_on_instance(self, z, x, **kwargs):
        raise NotImplementedError()

    def sample(self, bs, seed=None):
        raise NotImplementedError()

    def prepare_batch(self, batch):
        raise NotImplementedError()

    def train(self,
              itr,
              epochs,
              model_dir,
              result_dir,
              save_every=1,
              val_batch_size=None,
              scheduler_fn=None,
              scheduler_args={},
              verbose=True):
        for folder_name in [model_dir, result_dir]:
            if folder_name is not None and not os.path.exists(folder_name):
                os.makedirs(folder_name)
        f_mode = 'w' if not os.path.exists("%s/results.txt" % result_dir) else 'a'
        if val_batch_size is None:
            val_batch_size = itr.batch_size
        f = None
        if result_dir is not None:
            f = open("%s/results.txt" % result_dir, f_mode)
        for epoch in range(self.last_epoch, epochs):
            # Training
            epoch_start_time = time.time()
            if verbose:
                pbar = tqdm(total=len(itr))
            train_dict = OrderedDict({'epoch': epoch+1})
            for b, batch in enumerate(itr):
                if type(batch) not in [list, tuple]:
                    batch = [batch]
                batch = self.prepare_batch(batch)
                Z_batch = self.sample_z(batch[0].size()[0])
                losses, outputs = self.train_on_instance(Z_batch, *batch,
                                                         iter=b+1)
                for key in losses:
                    this_key = 'train_%s' % key
                    if this_key not in train_dict:
                        train_dict[this_key] = []
                    train_dict[this_key].append(losses[key])
                pbar.update(1)
                pbar.set_postfix(self._get_stats(train_dict, 'train'))
                # Process handlers.
                for handler_fn in self.handlers:
                    handler_fn(losses, [Z_batch] + batch, outputs,
                               {'epoch':epoch+1, 'iter':b+1, 'mode':'train'})
            if verbose:
                pbar.close()
            # Step learning rates.
            for key in self.scheduler:
                self.scheduler[key].step()
            all_dict = train_dict
            for key in all_dict:
                all_dict[key] = np.mean(all_dict[key])
            for key in self.optim:
                all_dict["lr_%s" % key] = \
                    self.optim[key].state_dict()['param_groups'][0]['lr']
            all_dict['time'] = \
                time.time() - epoch_start_time
            str_ = ",".join([str(all_dict[key]) for key in all_dict])
            print(str_)
            if f is not None:
                if (epoch+1) == 1:
                    # If we're not resuming, then write the header.
                    f.write(",".join(all_dict.keys()) + "\n")
                f.write(str_ + "\n")
                f.flush()
            if (epoch+1) % save_every == 0 and model_dir is not None:
                self.save(filename="%s/%i.pkl" % (model_dir, epoch+1),
                          epoch=epoch+1)

        if f is not None:
            f.close()

    def save(self, filename, epoch, legacy=False):
        dd = {}
        dd['g'] = self.g.state_dict()
        dd['d'] = self.d.state_dict()
        for key in self.optim:
            dd['optim_' + key] = self.optim[key].state_dict()
        dd['epoch'] = epoch
        torch.save(dd, filename)

    def load(self, filename, legacy=False, ignore_d=False):
        """
        ignore_d: if `True`, then don't load in the
          discriminator.
        """
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        dd = torch.load(filename,
                        map_location=map_location)
        self.g.load_state_dict(dd['g'])
        if not ignore_d:
            self.d.load_state_dict(dd['d'])
        for key in self.optim:
            if ignore_d and key == 'd':
                continue
            self.optim[key].load_state_dict(dd['optim_'+key])
        self.last_epoch = dd['epoch']
