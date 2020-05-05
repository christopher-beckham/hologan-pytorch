# hologan-pytorch

This repo is a loose reimplementation of HoloGAN, originally by Nguyen-Phuoc et al: https://arxiv.org/abs/1904.01326

I do not claim or guarantee any correctness of this implementation. This was implemented indepedently without consulting any of the original authors of the paper or other code.

## How to run

First, download the CelebA dataset, extract the images inside `img_align_celeba` to some directory, and export the environment variable `DATASET_CELEBA` to point to this folder (for instance, by running the command `export DATASET_CELEBA=/datasets/celeba/img_align_celeba`).

Then, run `python task_launcher.py`. To run the example training script, cd into `exps` and run `example.sh`.

Here is an example set of interpolations at 200 epochs.

<img src="https://github.com/christopher-beckham/hologan-pytorch/blob/dev/example_training.png?raw=true" width="800" />

## Notes

Here I bullet point some caveats and things that are perhaps worth noting if you are running HoloGAN experiments.

- The CelebA dataset isn't the exact same as what is presented in the paper, which appears to have background details cropped out from the images. Since that is not publicly available, here we simply use the original CelebA dataset resized down to 64px with no modifications.
- Contrary to what is described in the paper, I had trouble minimising the GAN loss and the identity regularisation term  (the z prediction term) when the z predictor was simply a _linear layer_ branching from the penultimate layer of the discriminator. In this code, both the real/fake predictor and the z predictor consist of a wide resblock branching from a base set of resblocks. Furthermore, while the paper proposes the z prediction term _only_ for high res generation (i.e. 128px), I strongly recommend you always use this term whenever you have more than two sources of inputs to a GAN (please consult the InfoGAN paper for more details), regardless of image resolution.
- The example script is simply a spectrally normalised JSGAN (i.e. binary x-entropy loss). You may get better quality with a hinge loss. Also, if you want to do higher res generation you'll have to modify the architecture scripts (TODO: I should get to this) and also try out the more recent GAN tricks, like equalised learning rates and larger batch sizes. 
