#!/bin/bash

cd ..

# Comment out if necessary.
export DATASET_CELEBA="/celebA/img_align_celeba"

export RESULTS_DIR="/results/hologan"


python task_launcher.py \
--dataset=cars \
--ngf=512 \
--ndf=64 \
--update_g_every=1  \
--lamb=25.0 \
--batch_size=32 \
--beta1=0.5 \
--beta2=0.999 \
--angles="[0,0,-90,90,0,0]" \
--resume=auto \
--name=example_experiment \
--trial_id=trial_01
