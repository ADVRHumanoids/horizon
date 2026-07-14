#!/bin/bash
set -e

# chown to user (needed after mounting volumes)
sudo chown user:user ~/test_ws ~/test_ws/src

# create and init ws
mkdir -p ~/test_ws
cd ~/test_ws
source ~/env/bin/activate
forest init

# env
source ~/scripts/env.bash

# add recipes
forest add-recipes git@github.com:advrhumanoids/multidof_recipes.git -t ros2

# hack: give unlimited rwx permissions to all users for the horizon folder
sudo chmod -R a+rwx ~/test_ws/src/horizon

# build
export PYTHONUNBUFFERED=1
forest grow horizon --verbose --clone-depth 1 -j ${FOREST_JOBS:-1}

# run basic import test
python -c "import horizon; import horizon.solvers.pyilqr"
