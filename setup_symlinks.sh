#!/bin/bash

## This script creates symlinks to outsource log folders. It requires a target directory as single
## parameter, e.g:
# bash setup_symlinks.sh $HOME/experiments/my-project

EXPERIMENTS_DIR=$1
[[ -z "$EXPERIMENTS_DIR" ]] && { echo "Error: No target directory was provided"; exit 1; }
DIRS=( "logs/runs" "logs/multiruns" "logs/experiments" "logs/debugs" "logs/evaluations" )

echo "symlink to $EXPERIMENTS_DIR..."
for d in "${DIRS[@]}"
do
    echo "symlink: $d"
    mkdir -p "$EXPERIMENTS_DIR/$d"
    ln -s -T "$EXPERIMENTS_DIR/$d" "$d"
done