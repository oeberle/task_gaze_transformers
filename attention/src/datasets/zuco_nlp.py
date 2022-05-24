from ..base.torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets.dataset import Dataset
from torchnlp.encoders.text import SpacyEncoder
from itertools import chain as datasets_iterator
# from torchnlp.utils import datasets_iterator
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_SOS_TOKEN
from torch.utils.data import Subset
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from ..utils.text_encoders import ClfBertProcessor, BaseProcessor
from ..utils.misc import clean_text
from .dataset_utils import compute_tfidf_weights, get_tokenizer_and_encoder
from torchtext import data, datasets
#from pytorch_pretrained_bert import BertTokenizer
from transformers import BertTokenizer

import numpy as np
import torch
import nltk
import pandas as pd
from dataloader.data_loader_zuco import ZucoLoader
from _local_options import zuco_files


def get_label_dict(task):
    
    if task in ['TSR', 'NR']: 
        label_list = ['AWARD', 'EDUCATION', 'EMPLOYER', 'FOUNDER', 'JOB_TITLE', 'NATIONALITY',
          'POLITICAL_AFFILIATION', 'VISITED', 'WIFE', 'CONTROL', None]
        # When do we need upper case labels here? -> convert to lower
        label_dict = {k:v.lower() if isinstance(v,str) else v for k,v in enumerate(label_list)}

        label_dict_inv = {v: k for k, v in label_dict.items()}
    elif task=='SR':
        label_dict = {-1: 'negative', 0: 'neutral', 1: 'positive'}
        
        label_dict_inv = {'negative':0, 'neutral':1, 'positive': 2}
    else:
        raise NotImplementedError
        
    return label_dict, label_dict_inv








class Zuco_NLP_Dataset(TorchnlpDataset):

    def __init__(self, root: str, task: str, tokenizer='spacy', use_tfidf_weights=False, append_sos=False,
                 append_eos=False, encoder=None, clean_txt=False, compute_tokenize_dict = False, exclude_control = False):
        super().__init__(root)

        self.n_classes = None
        self.task=task

        # Load dataset
        self.train_set, self.validation_set, self.test_set = zuco_dataset(train=True, test=True,
                                                                          validation=True, clean_txt=clean_txt,
                                                                          task=task, exclude_control=exclude_control,
                                                                          tokenizer=tokenizer)

        # Pre-process
        self.train_set.columns.add('index')
        self.validation_set.columns.add('index')
        self.test_set.columns.add('index')
        self.train_set.columns.add('weight')
        self.validation_set.columns.add('weight')
        self.test_set.columns.add('weight')
        self.train_set.columns.add('x')
        self.test_set.columns.add('x')
        self.validation_set.columns.add('x')

        # Subset train_set to normal class
        #self.train_set = Subset(self.train_set, train_idx_normal)

        # Make corpus and set encoder
        #text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.validation_set, self.test_set)]


        label_dict, label_dict_inv = get_label_dict(task)

        self.label_dict_inv = label_dict_inv #{v: k for k, v in label_dict.items()}
        
        if encoder is not None:
            # Set encoder to provided encoder
            self.encoder=encoder
            # We will only use Zuco_NLP_Dataset with a base encoder in place or for standard bert encoders
            pad_token_id = 0

        if tokenizer in ['bert', 'roberta-base']:
            assert encoder is None
           # tokenize = BertTokenizer.from_pretrained('bert-large-uncased',  do_lower_case=True)
           # self.encoder = ClfBertProcessor(tokenize, label2id=self.label_dict_inv, num_max_positions=256)

            tokenize, self.encoder = get_tokenizer_and_encoder(encoder, tokenizer, self.label_dict_inv, text_corpus=None)
            pad_token_id = int(tokenize.pad_token_id)
            

        self.padding_idx = pad_token_id
            
        k=0
        self.tokenize_dict = {}
        for row in datasets_iterator(self.train_set, self.validation_set, self.test_set):
            text = row['text']
            #import pdb;pdb.set_trace()
            row['text'], row['label'] = self.encoder.process_example((row['label'],row['text']))
            
            if compute_tokenize_dict==True:
                 
                ids2txt =self.encoder.tokenizer.decode(row['text']) if  tokenizer == 'bert' else  [self.encoder.tokenizer.vocab[i] for i in row['text']]
                
                
                self.tokenize_dict[k] = (text, row['text'], ids2txt) 
                # should be self.encoder.tokenizer.decode(row['text']) for bert
                k+=1
        
        # Compute tf-idf weights
        if use_tfidf_weights:
            compute_tfidf_weights(self.train_set, self.test_set, vocab_size=self.encoder.vocab_size)
        else:
            for row in datasets_iterator(self.train_set, self.validation_set, self.test_set):
                row['weight'] = torch.empty(0)

        # Get indices after pre-processing
        for i, row in enumerate(self.train_set):
            row['index'] = i
        for i, row in enumerate(self.validation_set):
            row['index'] = i
        for i, row in enumerate(self.test_set):
            row['index'] = i


        _ = self.create_word_map(idx_start=1 if tokenizer=='bert' else 0)
            

    def create_word_map(self, idx_start=0, idx_end=-1):
        self.word_mapping = {}

        for k,v in self.tokenize_dict.items():
            raw_text = v[0]
            for s in raw_text.split(' '):
                ids = [int(i) for i in self.encoder.tokenizer.encode(s)[idx_start:idx_end]]
                if s not in self.word_mapping:
                    try:
                        words = [self.encoder.tokenizer.vocab[i] for i in ids]
                    except:
                        words = [self.encoder.tokenizer.decode([i]) for i in ids]

                    self.word_mapping[s] = (ids, words)

        return None

    

def zuco_dataset(task=None, zuco_version=1, train=False, validation=False, test=False, clean_txt=False, exclude_control=False, tokenizer=None):

    ZM = ZucoLoader(zuco1_prepr_path=zuco_files, zuco2_prepr_path=None)
    df_human = ZM.get_zuco_task_df(zuco_version=zuco_version, task=task)

    if zuco_version==1:
        df_human = df_human[df_human.subject_id=='ZAB']
        ids = list(df_human.index)
        nid =len(ids)
    else:
        raise; print('Set reference subject_id.')
    
   # No need to shuffle only needed for evaluation, never for training 
   # np.random.seed(1)
   # np.random.shuffle(ids)
    
    sets = {}
    sets['train'], sets['test'], sets['validation'] = ids[:int(0.7*nid)], ids[int(0.7*nid):int((0.7+0.15)*nid)], ids[int((0.7+0.15)*nid):]
    
    
    splits = [split_set for (requested, split_set) in [(train, 'train'), (validation, 'validation'), (test, 'test')] if requested]
    
    ret = []        
        
    for split_set in splits:
        
        examples = []

        for id_ in sets[split_set]:
            row = df_human.loc[id_]


            if clean_txt:
                text = clean_text(row.words)
            else:
                text = ' '.join(row.words)

            if 'label' in df_human.columns:
                label = row.label
            elif 'labels' in df_human.columns:
                label = row.labels 
            else:
                raise
                
                
            if isinstance(label, int):
                
                label_dict,_ =  get_label_dict(task)
                label = label_dict[label]
                
            if label=='CONTROL':
                print('Skipping TSR CONTROL condition for now')
                continue
                        
            examples.append({
                'text':  str(text) if tokenizer in ['roberta-base', 't5-base'] else  str(text).lower(),
                'label': label.lower(),
                'raw': row.words,
                'text_id': row.text_id
            })

        ret.append(Dataset(examples))

        
    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
    