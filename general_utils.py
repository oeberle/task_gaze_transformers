import os
import pandas as pd
import yaml

import models
import torch as tr
import torch.nn as nn
from attention.src.networks.main import build_embedding

from optimizer_utils import *
from models import BertModel, RobertaModel, T5Model
import re
from os import path

all_subjects = ['ZGW',
                'ZJN',
                'ZJS',
                'ZDM',
                'ZJM',
                'ZKH',
                'ZKB',
                'ZMG',
                'ZPH',
                'ZAB',
                'ZKW']

all_subjects2 = ['YSL',
                 'YAC',
                 'YIS',
                 'YDG',
                 'YTL',
                 'YMD',
                 'YLS',
                 'YAK',
                 'YHS',
                 'YRH',
                 'YRK',
                 'YFR',
                 'YAG',
                 'YFS',
                 'YDR',
                 'YMS',
                 'YRP',
                 'YSD']

label_dict_str = {'award': 0, 'education': 1, 'employer': 2, 'founder': 3, 
              'job title': 4, 'nationality': 5, 
              'political affiliation': 6, 'visited': 7, 'wife': 8}


label_dict_sr = {'negative':0,'neutral':1, 'positive':2}


def get_analysis_config(task, analysis_file=None):
    analysis_dir = path.abspath('configs/analysis')
    if analysis_file is None:
        if task == 'TSR':
            analysis_file = 'wikirel_base_pub.yaml'
        elif task == 'SR':
            analysis_file = 'sst_base_pub.yaml'
        else:
            raise NotImplementedError

    with open(path.join(analysis_dir, analysis_file), 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    return cfg


def set_up_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_cfg(config_path):            
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg
            
def construct_df_from_single(df_dir, job_file):
    job_dict = pd.read_pickle(job_file)
    keys = sorted(job_dict.keys())

    df_list = []
    for ind in keys:
        job = job_dict[ind]
        print(job)

        df_file = df_dir.format(job[-1])
        df_ = pd.read_pickle(df_file)
        df_list.append(df_)

    df_out = pd.concat(df_list, axis=0)
    res_file = df_dir.replace('_temp', '').replace('_{}.pkl', '.pkl')
    
    assert len(df_out) == len(jobs)
    df_out.to_pickle(res_file)
    print(res_file)
    return None


def load_state_dict(model, cfg, model_name, model_folder):
    
    if 'bnc_freq' in model_name:
        pass    
    else:
        weight_file = os.path.join(model_folder,'{}-model.pt'.format(model_name))
        try:
            state_dict = tr.load(weight_file)
        except:
            state_dict = tr.load(weight_file, map_location=tr.device('cpu'))
            
            
        if 'roberta' in model_name:
            state_dict = {k.replace('module.','roberta.'):v for k,v in state_dict.items()}
            
        elif 'bert'  in model_name:
            state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
            
        elif 't5' in model_name:
            pass
            
        model.load_state_dict(state_dict)
        model.eval()
        print('Initialized {} from {}'.format(model_name, weight_file))


def load_model_from_cfg(cfg, model_name, output_dim ,training_dataset, dataset_base, device):
    
    OUTPUT_DIM = output_dim
    if 'bert' in model_name or 't5' in model_name:
        pretrained_embeddings = None  
        
    else:
        pretrained_embeddings = build_embedding(dataset=training_dataset,
                                                embedding_size=cfg['embedding_size'],
                                                pretrained_model=cfg['pretrained_model'],
                                                update_embedding=False,
                                                embedding_reduction=cfg['embedding_reduction'],
                                                use_tfidf_weights=cfg['use_tfidf_weights'],
                                                normalize_embedding=cfg['normalize_embedding'],
                                                word_vectors_cache=cfg['word_vectors_cache'])

        INPUT_DIM = training_dataset.encoder.tokenizer.vocab_size
        EMBEDDING_DIM = cfg['embedding_size']
        N_FILTERS = cfg['n_filters']
        FILTER_SIZES = cfg['filter_sizes']
        DROPOUT = cfg['dropout']
        PAD_IDX = training_dataset.encoder.tokenizer.padding_index
        MASK_IDX = dataset_base.encoder.tokenizer.token_to_index['<mask>']
        EOS_IDX = training_dataset.encoder.tokenizer.eos_index
        UNK_IDX = training_dataset.encoder.tokenizer.unknown_index
        
        
    if 'cnn_1fs' in model_name:
        FILTER_SIZES = FILTER_SIZES[0] # for now only ONE filter size
        model = models.Kim2014CNN_1FS(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES,
                                      OUTPUT_DIM, DROPOUT, PAD_IDX)
    elif 'cnn_mfs' in model_name:
        model = models.Kim2014CNN_MFS(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES,
                                      OUTPUT_DIM, DROPOUT, PAD_IDX)
    elif 'weighted_boe' in model_name:
        model = models.WeightedBOE(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES,
                                   OUTPUT_DIM, DROPOUT, [PAD_IDX, UNK_IDX], PAD_IDX, OUTPUT_DIM)
    elif 'self_attention' in model_name:
        N_ATTENTION_HEADS = cfg['n_attention_heads']
        model_name = model_name #+ '_'+ str(N_ATTENTION_HEADS)
        COMPUTE_R = False
        model = models.SelfAttentionModel(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES,
                                          OUTPUT_DIM, DROPOUT, [PAD_IDX, UNK_IDX], PAD_IDX,
                                          OUTPUT_DIM, N_ATTENTION_HEADS, COMPUTE_R)
        
    elif 'bnc_freq' in model_name:
        model = models.BNCFrequencies(encoder=dataset_base.encoder )
        
    elif 'roberta-base' in model_name:
        model = RobertaModel(OUTPUT_DIM,  cfg['pretrained_model'])
 
    elif 'bert' in model_name:
        model = BertModel(OUTPUT_DIM, cfg['pretrained_model'])
        
    elif 't5' in model_name:
        model = T5Model(OUTPUT_DIM, cfg['pretrained_model'])
        
    else:
        raise ValueError("model not known.")

    model.model_name = model_name
    model.device=device


    return model, pretrained_embeddings

def _condense_text(text:str, CONDENSER_PATTERN = "[\W1]"):
    condensed = re.sub(CONDENSER_PATTERN, "", text)
    condensed = condensed.lower()
    return condensed
