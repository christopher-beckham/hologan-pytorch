import glob
import sys
import os
import argparse
from hologan import HoloGAN
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from iterators.datasets import CelebADataset
from torchvision.transforms import transforms
from architectures import arch
from collections import OrderedDict
from tools import (count_params)

use_shuriken = False
try:
    from shuriken.utils import get_hparams
    use_shuriken = True
except:
    pass

# This dictionary's keys are the ones that are used
# to auto-generate the experiment name. The values
# of those keys are tuples, the first element being
# shortened version of the key (e.g. 'dataset' -> 'ds')
# and a function which may optionally shorten the value.
id_ = lambda x: str(x)
kwargs_for_name = {
    'arch': ('arch', lambda x: os.path.basename(x)),
    'batch_size': ('bs', id_),
    'ngf': ('ngf', id_),
    'ndf': ('ndf', id_),
    'lr': ('lr', id_),
    'lamb': ('lamb', id_),
    'angles': ('angles', lambda x: x[1:-1].replace(",","_")),
    'update_g_every': ('g', id_),
    'beta1': ('b1', id_),
    'beta2': ('b2', id_),
    'use_64px': ('use_64px', id_),
    'trial_id': ('_trial', id_),
    'z_extra_fc': ('dxz', id_)
}

def generate_name_from_args(dd):
    buf = {}
    for key in dd:
        if key in kwargs_for_name:
            if dd[key] is None:
                continue
            new_name, fn_to_apply = kwargs_for_name[key]
            new_val = fn_to_apply(dd[key])
            if dd[key] is True:
                new_val = ''
            buf[new_name] = new_val
    buf_sorted = OrderedDict(sorted(buf.items()))
    #tags = sorted(tags.split(","))
    name = ",".join([ "%s=%s" % (key, buf_sorted[key]) for key in buf_sorted.keys()])
    return name


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--angles', type=str, default="[0,0,0,360,0,0]")
    parser.add_argument('--use_64px', action='store_true')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--nmf', type=int, default=32)
    parser.add_argument('--nb', type=int, default=2) # num blocks
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--z_extra_fc', action='store_true')
    parser.add_argument('--lamb', type=float, default=0.)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--update_g_every', type=int, default=5)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--save_images_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default='auto')
    parser.add_argument('--trial_id', type=str, default=None)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

args = parse_args()
args = vars(args)

print("args before shuriken:")
print(args)

if use_shuriken:
    shk_args = get_hparams()
    print("shk args:", shk_args)
    # Stupid bug that I have to fix: if an arg is ''
    # then assume it's a boolean.
    for key in shk_args:
        if shk_args[key] == '':
            shk_args[key] = True
    args.update(shk_args)

if args['trial_id'] is None and 'SHK_TRIAL_ID' in os.environ:
    print("SHK_TRIAL_ID found so injecting this into `trial_id`...")
    args['trial_id'] = os.environ['SHK_TRIAL_ID']

if args['name'] is None and 'SHK_EXPERIMENT_ID' in os.environ:
    print("SHK_EXPERIMENT_ID found so injecting this into `name`...")
    args['name'] = os.environ['SHK_EXPERIMENT_ID']

name = generate_name_from_args(args)

torch.manual_seed(args['seed'])

# This is the one from the progressive growing GAN
# code.
IMG_HEIGHT = 64 if args['use_64px'] else 32
train_transforms = [
    transforms.Resize(IMG_HEIGHT),
    transforms.CenterCrop(IMG_HEIGHT),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

ds = CelebADataset(root=os.environ['DATASET_CELEBA'],
                   transforms_=train_transforms)
loader = DataLoader(ds,
                    batch_size=args['batch_size'],
                    shuffle=True,
                    num_workers=args['num_workers'])

if args['save_path'] is None:
    args['save_path'] = os.environ['RESULTS_DIR']

gen, disc = arch.get_network(args['z_dim'],
                             ngf=args['ngf'],
                             ndf=args['ndf'],
                             use_64px=True if args['use_64px'] else False,
                             z_extra_fc=args['z_extra_fc'])

print("Generator:")
print(gen)
print(count_params(gen))
print("Disc:")
print(disc)
print(count_params(disc))


angles = eval(args['angles'])
gan = HoloGAN(
    gen_fn=gen,
    disc_fn=disc,
    z_dim=args['z_dim'],
    lamb=args['lamb'],
    angles=angles,
    opt_d_args={'lr': args['lr_d'], 'betas': (args['beta1'], args['beta2'])},
    opt_g_args={'lr': args['lr_g'], 'betas': (args['beta1'], args['beta2'])},
    update_g_every=args['update_g_every'],
    handlers=[]
)

def _image_handler(gan, out_dir, batch_size=32):
    def _image_handler(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1:
            if kwargs['epoch'] % args['save_images_every'] == 0:
                gan._eval()
                mode = kwargs['mode']
                if mode == 'train':
                    # TODO: do for valid as well
                    epoch = kwargs['epoch']
                    z_batch = gan.sample_z(batch_size)
                    z_batch = z_batch.cuda()
                    for key in ['yaw', 'pitch', 'roll']:
                        rot = gan._generate_rotations(z_batch,
                                                      min_angle=gan.angles['min_angle_%s' % key],
                                                      max_angle=gan.angles['max_angle_%s' % key],
                                                      axes=[key],
                                                      num=20)
                        #padding = torch.zeros_like(rot['yaw'][0])+0.5
                        save_image( torch.cat(rot[key], dim=0),
                                    nrow=batch_size,
                                    filename="%s/rot_%s_%i.png" % (out_dir, key, epoch) )

    return _image_handler


if args['name'] is None:
    save_path = "%s/s%i" % (args['save_path'], args['seed'])
else:
    save_path = "%s/s%i/%s" % (args['save_path'], args['seed'], args['name'])
if not os.path.exists(save_path):
    os.makedirs(save_path)


expt_dir = "%s/%s" % (save_path, name)
if not os.path.exists(expt_dir):
    os.makedirs(expt_dir)

gan.handlers.append(_image_handler(gan, expt_dir))

print("expt_dir:", expt_dir)

if args['resume'] is not None:
    if args['resume'] == 'auto':
        # autoresume
        # List all the pkl files.
        files = glob.glob("%s/*.pkl" % expt_dir)
        # Make them absolute paths.
        files = [os.path.abspath(key) for key in files]
        if len(files) > 0:
            # Get creation time and use that.
            latest_model = max(files, key=os.path.getctime)
            print("Auto-resume mode found latest model: %s" %
                  latest_model)
            gan.load(latest_model)
    else:
        print("Loading model: %s" % args['resume'])
        gan.load(args['resume'])

if args['interactive']:

    bs = 32
    gan._eval()

    z_batch = gan.sample_z(bs, seed=None)
    if gan.use_cuda:
        z_batch = z_batch.cuda()

    import numpy as np

    # -45 to +45 deg
    rot = gan._generate_rotations(z_batch,
                                  min_angle=-2*np.pi,
                                  max_angle=2*np.pi,
                                  num=50)
    padding = torch.zeros_like(rot['yaw'][0])+0.5

    imgs = torch.cat(rot['yaw'] + \
                     [padding] + \
                     rot['pitch'] + \
                     [padding] + \
                     rot['roll'], dim=0)

    save_image( imgs,
                nrow=bs,
                filename="%s/%s/gen_z.png" % (args['save_path'],
                                              args['name']))



    #import pdb
    #pdb.set_trace()

else:
    gan.train(itr=loader,
              epochs=args['epochs'],
              model_dir=expt_dir,
              result_dir=expt_dir,
              save_every=args['save_every'])
