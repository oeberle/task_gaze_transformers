#!/bin/bash -e
# Script to prepare text to the same pre-processing as in the 1 Billion words corpus
# Assumes repository https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark has been cloned

# locale settings.
export LANG=en_US.UTF-8
export LANGUAGE=en_US:
export LC_ALL=en_US.UTF-8


src_file='input.txt'
target_file='input_proc.txt'

echo "Working on $src_file"
time cat $src_file | \
  /1-billion-word-language-modeling-benchmark/scripts/normalize-punctuation.perl -l en | \
  /1-billion-word-language-modeling-benchmark/scripts/tokenizer.perl -l en > \
    $target_file
    
echo "Done working on $src_file"