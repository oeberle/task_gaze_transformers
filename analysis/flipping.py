import pandas as pd
import numpy as np
import pickle
import yaml
import re
import itertools 
import os
import matplotlib.pyplot as plt
import torch as tr
from eval_utils import *
import click
import glob
from analysis.flip_utils import load_trained_model, flip

def flipping(dfs_pfile, 
             config_file, 
             task,  
             cases,
             cases_to_flip = ['tsr','bnc_freq_prob', 'cnn0.50', 'sattn_rels_0.25', 
                              'base_bert_flow_11', 'base_roberta_flow_11', 'base_t5_flow_11', 'random'],
            ignore_filt = [1, -1], fracs = np.linspace(0.,1.,21) ):
    
    """
    Extract input reduction (token flipping) curves.

    Args:
       
        dfs_pfile: Output dataframe of run_alignment.py that contains sentences and attributions for each model
        config_file: Training config file for the model used for the reduction experiment.
        task:  "SR" or "TSR" for sentiment reading (SST) and task-specific reading (Wikipedia)
        cases: All analysis cases that were considered during alignment.
        cases_to_flip: Model cases that are used for the input reduction experiments.
        ignore_filt: Mask to apply over the sentence.

    Returns
        dict: Dictionary contains the input reduction results incl curves and reduction order.
    """
    
    
    dfs_all = pickle.load(open(dfs_pfile, 'rb'))

    if 'bert' in config_file:
        ref_case_ = [k for k in dfs_all.keys() if 'bert' in k][0]
        UNK_IDX = 100
        EOS = '[SEP]'
    else:
        print('Please add another if-block for the use of other models than "bert-x"')
        raise

    remove_list, include_list, extra_mapping, allowed_matches = get_matching_vars(ref_case_, task)

    print('Using ref case for input reduction', ref_case_)
    print('Loading weights for input reduction from', config_file)

    if 'bert' in ref_case_:
        device = [tr.device('cuda:{}'.format(idevice))
                  for idevice in range(tr.cuda.device_count())][0] if tr.cuda.is_available() else ['cpu']
    else:
        device = ['cpu']

    model = load_trained_model(config_file, device=device)


    all_flips_cases  =  {}

    for flip_case in ['generate']: 
        all_flips = {}

        skip=0
        for case in cases_to_flip: 

            random_order = True if case=='random' else False      
            if random_order==True:
                random_order = True
                case_ = ref_case_
                x_temp_ref = 'x'
            else:
                random_order = False
                case_ = case
                x_temp_ref = [c for c in cases if c[0]==case][0][1][1]

                if x_temp_ref == 'x_abs':
                    print('Use signed relevance values for perturbation', case)
                    x_temp_ref = 'x'

            M,E, EVOLUTION = [],[], []
            j = 0
            for ix, row in dfs_all[case_].iterrows():
                ref_row = dfs_all[ref_case_].iloc[j]
                
                try:
                    m, e, evolution = flip(model, row, ref_row, x_temp_ref = x_temp_ref, fracs=fracs,
                                           flip_case=flip_case,  UNK_IDX = UNK_IDX, random_order=random_order,
                                           allowed_matches = allowed_matches, extra_mapping = extra_mapping,
                                           remove_list=remove_list, include_list=include_list, EOS=EOS,
                                           ref_model_name=ref_case_, ignore_filt=ignore_filt)

                except:
                    import pdb;pdb.set_trace()
                

                if m is None:
                    print('Skipping flip', ref_row)
                    skip+=1
                    j+=1
                    continue
                    
                M.append(m)
                E.append(e)
                EVOLUTION.append(evolution)
                j+=1

            print(flip_case, case, skip)
            all_flips[case]= {'E':E, 'M':M, 'Evolution':EVOLUTION}    

        all_flips_cases[flip_case] = all_flips

    return all_flips_cases
