#!/bin/bash
set -e

# chown to user (needed after mounting volumes)
sudo chown user:user ~/test_ws ~/test_ws/src

# create and init ws
mkdir -p ~/test_ws
cd ~/test_ws
forest init

# env
source ~/scripts/env.bash

# add recipes
forest add-recipes git@github.com:advrhumanoids/multidof_recipes.git -t ros2

# build
export PYTHONUNBUFFERED=1
forest grow horizon --verbose --clone-depth 1 -j ${FOREST_JOBS:-1}

