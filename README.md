# hologan-pytorch

This repo is a loose reimplementation of HoloGAN, originally by Nguyen-Phuoc et al: https://arxiv.org/abs/1904.01326

I do not claim or guarantee any correctness of this implementation. This was implemented indepedently without consulting any
of the original authors of the paper or other code (though as of time of writing, I was not able to find any other implementation
of this on Github).

There are two issues I am trying to iron out (see my 'notes' section below):
- Stability of the identity regulariser term (this might require a reformulation of the discriminator network to have a deeper/wider branch for the z prediction)
- Oscilliating rotations (i.e. left-right-left-right behaviour, rather than left-right)
- Official code uses trilinear sampling for rotation. PyTorch only has bilinear for 3D. TODO add trilinear, though it is slow since it's not a C/CUDA implementation.

## How to run

First, download the CelebA dataset, extract the images inside `img_align_celeba` to some directory, and export the environment 
variable `DATASET_CELEBA` to point to this folder (for instance, by running the command `export DATASET_CELEBA=/datasets/celeba/img_align_celeba`). (Note: this is not the same as the CelebA in the paper, which seems to have undergone some post-processing to crop background. The scripts to generate this dataset are not available in the official HoloGAN repo.)

Then, run `python task_launcher.py`. To run an example training script, cd into `exps` and run `test.sh`. For the sake of
convenience, I have also pasted the contents of the script here:

```
#!/bin/bash

cd ..

# either use the `--save_path` arg
# or export this environment variable
export RESULTS_DIR="/results/hologan"

python task_launcher.py \
--ngf=256 `#num filters for generator` \
--ndf=128 `#num filters for disc` \
--update_g_every=1 `#update g every this many iterations`  \
--lamb=1.0 `#coefficient for identity reg loss` \
--resume=auto `#find latest checkpoint and resume, if it exists` \
--name="official_test_01" `#experiment name` \
--trial_id="TEST" `#unique identifier for experiment`
```

On my setup, the generated experiment folder corresponds to: 

`/results/hologan/s0/official_test_01/_trial=TEST,angles=0_0_0_360_0_0,b1=0.0,b2=0.999,bs=32,dxz=False,g=1,lamb=1.0,ndf=128,ngf=256,use_64px=False`

Essentially, experiments are structured like so:

`<results_dir>/<seed>/<experiment_name>/_trial=<trial_id>,flag1=val1,flag2=val2,...,flagn_valn`

## Notes

**12/09/2019**

As of time of writing, I have had some trouble implementing the identity regularisation term in section 3.3 of the paper:

<p align="center">
<img src="https://user-images.githubusercontent.com/2417792/64811185-c7644e00-d56a-11e9-813c-602e76099d2a.png" width="500" />
</p>

In particular, with a sufficiently parameterised discriminator (e.g. `ndf=128`) and various coefficients for this 
reconstruction loss,  the GAN loss inevitably gets worse over time, which results in poorer sample quality (see plot directly 
below). It also takes quite a while for the z loss to decrease too, and this is simply for 32x32 CelebA. I also ran
experiments where an entire new discriminator-like network is used to predict z (rather than branching off D), but I vaguely
recall this not making a difference.

<p align="center">
<img src="https://user-images.githubusercontent.com/2417792/64811358-190cd880-d56b-11e9-9a18-2b87933bfbde.png" width=600 />

(Figure 1: reconstruction loss for identity regulariser term. It goes down, albeit quite slowly...)
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/2417792/64811533-7acd4280-d56b-11e9-8ea2-6eef92435828.png" width=600 />

(Figure 2: generator loss. It goes up over time, which means we would expect sample quality to become more abysmal over time)
</p>

Here are some images from one of these experiments:

<p align="center">
<img src="https://user-images.githubusercontent.com/2417792/64812202-b9afc800-d56c-11e9-9698-3823772ae6f3.png" width=600 />

(Figure 3. Sampling is only done on the z axis (yaw), and angles are sampled between -60 and 60 degrees, which seems to roughly
correspond to the range of poses observed on that axis for the dataset)
</p>

What we can also see is that the rotations oscillate from left-right-left rather than smoothly transitioning from left to right (and vice versa). The original paper shows the latter kind of oscillation, and currently it's not clear to me why I'm not able to achieve it (or if it's even easy to accomplish in an unsupervised setting).
