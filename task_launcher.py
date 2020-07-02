import glob
import sys
import os
import argparse
from hologan import HoloGAN
import torch
import numpy as np
import tempfile
import random
import string
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from iterators.datasets import (CelebADataset,
                                CarsDataset)
from torchvision.transforms import transforms
from architectures import arch
from collections import OrderedDict
from tools import (count_params, generate_rotations)

use_shuriken = False
try:
    # This only applies to me. If you're not me,
    # don't worry about this code.
    from shuriken.utils import get_hparams
    use_shuriken = True
except:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='celeba',
                        choices=['celeba', 'cars'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--angles', type=str, default="[0,0,-45,45,0,0]",
                        help="""
                        A string that should eval() into a list of 6
                        values corresponding to the min/max for sampling
                        (uniformly) degree values from axes x, y, and z,
                        respectively.
                        (Note that the 'y' axis here is the one pointing
                        up/down, denoting the yaw.)
                        """)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--nmf', type=int, default=32)
    parser.add_argument('--nb', type=int, default=2) # num blocks
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--z_dim', type=int, default=128)
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

if use_shuriken:
    # This only applies to me. If you're not me,
    # don't worry about this code.
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
else:
    if args['trial_id'] is None:
        print("trial_id not defined so generating random id...")
        trial_id = "".join([ random.choice(string.ascii_letters[0:26]) for j in range(5) ])
        args['trial_id'] = trial_id

if 'SHK_EXPERIMENT_ID' in os.environ:
    print("SHK_EXPERIMENT_ID found so injecting this into `name`...")
    args['name'] = os.environ['SHK_EXPERIMENT_ID']
else:
    if args['name'] is None:
        raise Exception("You must give a name to this experiment")

torch.manual_seed(args['seed'])

IMG_HEIGHT = 64
train_transforms = [
    transforms.Resize(IMG_HEIGHT),
    transforms.CenterCrop(IMG_HEIGHT),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

if args['dataset'] == 'celeba':
    ds = CelebADataset(root=os.environ['DATASET_CELEBA'],
                       transforms_=train_transforms)
else:
    ds = CarsDataset(root=os.environ['DATASET_CARS'],
                     transforms_=train_transforms)
loader = DataLoader(ds,
                    batch_size=args['batch_size'],
                    shuffle=True,
                    num_workers=args['num_workers'])

if args['save_path'] is None:
    args['save_path'] = os.environ['RESULTS_DIR']

gen, disc = arch.get_network(z_dim=args['z_dim'],
                             ngf=args['ngf'],
                             ndf=args['ndf'])

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
                    for key in ['x', 'y', 'z']:
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

save_path = "%s/s%i/%s" % \
    (args['save_path'], args['seed'], args['name'])
if not os.path.exists(save_path):
    os.makedirs(save_path)


expt_dir = "%s/%s" % (save_path, args['trial_id'])
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

    for axis in ['y']:
        print("Generating frames for axis %s..." % axis)

        tmp_dir = tempfile.mkdtemp()
        print("Temp dir: %s" % tmp_dir)

        out_mp4_dir = "%s/%s/%s" % \
            (args['save_path'], args['name'], axis)
        print("Destination dir for mp4: %s" % out_mp4_dir)

        if not os.path.exists(out_mp4_dir):
            os.makedirs(out_mp4_dir)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        generate_rotations(gan,
                           z_batch,
                           tmp_dir,
                           axis,
                           num=500)

        # Remove old mp4 file if it exists.
        if os.path.exists("%s/out.mp4" % out_mp4_dir):
            os.remove("%s/out.mp4" % out_mp4_dir)

        from subprocess import check_output
        fps = 48
        crf = 4

        ffmpeg_out = check_output(
            "cd %s; ffmpeg -framerate %i -pattern_type glob -i '*.png' -crf %i -c:v libx264 out.mp4" % (tmp_dir, fps, crf),
            shell=True)
        ffmpeg_out = ffmpeg_out.decode('utf-8').rstrip()
        print(ffmpeg_out)

        copy_out = check_output(
            "cp %s/out.mp4 %s/out.mp4" % (tmp_dir, out_mp4_dir),
            shell=True
        )
        print(copy_out)

else:
    gan.train(itr=loader,
              epochs=args['epochs'],
              model_dir=expt_dir,
              result_dir=expt_dir,
              save_every=args['save_every'])
