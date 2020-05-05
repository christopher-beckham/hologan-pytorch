# hologan-pytorch

This repo is a loose reimplementation of HoloGAN, originally by Nguyen-Phuoc et al: https://arxiv.org/abs/1904.01326

I do not claim or guarantee any correctness of this implementation. This was implemented indepedently without consulting any of the original authors of the paper or other code (though as of time of writing, I was not able to find any other implementation of this on Github).

## How to run

First, download the CelebA dataset, extract the images inside `img_align_celeba` to some directory, and export the environment 
variable `DATASET_CELEBA` to point to this folder (for instance, by running the command `export DATASET_CELEBA=/datasets/celeba/img_align_celeba`).

Then, run `python task_launcher.py`. To run the example training script, cd into `exps` and run `example.sh`.

## Notes


