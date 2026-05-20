#!/bin/bash
set -e

# get platform name from argument
PLATFORM="$1"

# error if no argument
if [ -z "$PLATFORM" ]; then
  echo "Usage: $0 <platform>"
  exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/$PLATFORM

docker compose exec dev ./scripts/test.bash