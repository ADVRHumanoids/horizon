#!/bin/bash
set -ev

# required packages for building horizon
FOREST_ARGS="--default-build-type Release --clone-protocol https --verbose -j2"
cd $FOREST_DIR

# required due to shell processes (that spawn independently every .sh file)!
source setup.bash

forest grow pybind11 $FOREST_ARGS
forest grow horizon $FOREST_ARGS

# run tests
./travis/test.sh

# upload if push is tag
if [ -z $TRAVIS_TAG ]; then 
    echo "Not a tag build, will not upload to pypi"; 
else 
    python3 -m build --wheel && twine upload -u __token__ -p $PYPI_TOKEN dist/*.whl;
fi

DOCKER_UPLOAD=false

if [ "$DOCKER_UPLOAD" = true ]; then
# upload if push is tag
    if [ -z $TRAVIS_TAG ]; then 
        echo "Not a tag build, will not upload to docker"; 
    else 
        cd $HORIZON_DIR/docker && ./upload.sh;
    fi
fi