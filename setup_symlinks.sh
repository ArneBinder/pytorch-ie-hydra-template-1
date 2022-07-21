#!/bin/bash

## Log and model folders can grow very large. This script creates symlinks to outsource them.
## It requires a target directory as single parameter, e.g:
##
## bash setup_symlinks.sh $HOME/experiments/my-project
##
## Note, that training models with a different name than "default" may result in a different
## save_dir (see config/train.yaml) which needs to be symlinked manually, if outsourcing them
## is intended.

EXPERIMENTS_DIR=$1
[[ -z "$EXPERIMENTS_DIR" ]] && { echo "Error: No target directory was provided"; exit 1; }
DIRS=( "logs/experiments" "logs/evaluations" "logs/debugs" "models/default" "models/conll2003" )

echo "symlink to $EXPERIMENTS_DIR..."
for d in "${DIRS[@]}"
do
    echo "symlink: $d"
    mkdir -p "$EXPERIMENTS_DIR/$d"
    ln -s -T "$EXPERIMENTS_DIR/$d" "$d"
done
