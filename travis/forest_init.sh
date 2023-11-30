#!/bin/bash -i
pip3 install build twine hhcm-forest==1.1.14
SRC_FOLDER=$PWD
cd .. && mkdir forest_ws && cd forest_ws && forest init  # create forest ws for building
# source setup.bash
ln -s $SRC_FOLDER src/$(basename $SRC_FOLDER)  # symlink original source folder

# moving the recipes manually defined so as to freeze this build configuration
echo "moving build recipes to $PWD"
mv ../horizon/travis/recipes .