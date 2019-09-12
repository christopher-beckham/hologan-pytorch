#!/bin/bash

# Run this from root directory, *not* from
# the dev folder.

HOLO_DIR=../hologan/


cp $HOLO_DIR/{hologan.py,base.py,tools.py} .

cp -r $HOLO_DIR/docker docker

cp -r $HOLO_DIR/architectures architectures
cp -r $HOLO_DIR/iterators iterators
