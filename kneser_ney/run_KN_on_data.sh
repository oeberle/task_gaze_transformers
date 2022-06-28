#!/bin/bash

srilm_bin="/home/oeberle/srilm/lm/bin/i686-m64"
order=5

# Assumes an already trained KN model, here on the 1 Billion corpus
model_dir="${order}gram"

output_dir="output_KN"

mkdir $output_dir

test_set="input_proc.txt"

ngram="$srilm_bin/ngram"
ngramcount="$srilm_bin/ngram-count"
model="KneserNeyalaChenGoodmanInterpolated" 
flag="-kndiscount -interpolate"

common_ppl_flags="-unk -order $order -debug 2"
common_ppl_flags_summary="-unk -order $order -debug 0"


# Evaluating perplexity
$ngram $common_ppl_flags -lm $model_dir/${model}.gz -ppl $test_set > $output_dir/log.$model.ppl 2>&1
$ngram $common_ppl_flags_summary -lm $model_dir/${model}.gz -ppl $test_set > $output_dir/log.summary$model.ppl 2>&1
