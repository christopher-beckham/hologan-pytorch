#!/bin/bash

# This must be run from the ROOT directory of
# this repo, NOT inside the docker folder!!!

nvidia-docker build -t beckhamc1 -f docker/Dockerfile .
