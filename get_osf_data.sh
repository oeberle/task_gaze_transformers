#!/bin/bash

# Requires: pip install osfclient

cwd=$(pwd)
cd data/proc
   
# Get data from task_tuned_transformers project (https://osf.io/q5afk/)
 osf -p q5afk clone

cd q5afk/osfstorage

unzip "*.zip"

cd $cwd

mkdir data/eval_files

# Collect all evaluation result *.pkl files that contain the different model and human attributions
find ./ -name '*.pkl' -exec cp -prv '{}' 'data/eval_files' ';'