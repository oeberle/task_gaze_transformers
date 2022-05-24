from analysis.flipping import flipping
from general_utils import set_up_dir
import click
from _local_options import eval_folder, model_output, zuco_files
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml

@click.command()
@click.option('--df_all_file')
@click.option('--analysis_file')
@click.option('--config_file')
@click.option('--task')

def main(df_all_file, analysis_file, config_file, task): 
    """
    Function to compute the flipping experiments

    Args:
        df_all_file (str): Filename of the output dataframe of run_alignment.py that 
        contains sentences and attributions for each model.
        analysis_file (str): Filename of the analysis yaml, see configs/analysis/*.yaml
        config_file (str): Filename of the model used for the input reduction experiments, see the *.yaml file hat contains the model checkpoint
        task (str): {"TSR", "SR"}

    Returns:
        None
    """
    
    
    out_dir = eval_folder
    
    out_dir_flip = os.path.join(out_dir, 'input_reduction')
    set_up_dir(out_dir_flip)
    
    
    with open(analysis_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    cases = [(k, v) for k, v in cfg.items()] 

    all_flips_cases = flipping(df_all_file, 
                     config_file, 
                     task,
                     cases,
                     cases_to_flip = ['tsr', 'ez_nr','bnc_freq_prob', 
                                      'cnn0.50', 'sattn_rels_0.25', 
                                      'base_bert_flow_11', 'base_roberta_flow_11', 
                                      'base_t5_flow_11', 'random']
                      )

    pickle.dump(all_flips_cases, open(os.path.join(out_dir_flip, 'all_flip_cases_{}.p'.format(task)), 'wb'))



if __name__=='__main__':    
    main()