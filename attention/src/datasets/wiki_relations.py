import pandas as pd
import os
import torch as tr

from torchnlp.datasets.dataset import Dataset
from itertools import chain as datasets_iterator
from torchnlp.encoders.text import SpacyEncoder
from ..utils.text_encoders import BaseProcessor
from ..base.torchnlp_dataset import TorchnlpDataset
import pickle
import itertools
from general_utils import _condense_text
import dataloader.data_loader_zuco as dlz
from glob import glob
import numpy as np
import collections


from ..utils.text_encoders import ClfBertProcessor, BaseProcessor
from transformers import BertTokenizer

from dataloader.data_loader_zuco import ZucoLoader
from .dataset_utils import get_tokenizer_and_encoder, compute_tfidf_weights

from _local_options import wikirel_folder, zuco_files







def get_subset(dataset, valid_label_dict):
    data_subset = [c for i,c in enumerate(dataset) if c['label'] in valid_label_dict.keys()]
    dataset_filtered = Dataset(data_subset)
    return dataset_filtered


class WikiRelations(TorchnlpDataset):

    def __init__(self, tokenizer='spacy', encoder=None, zuco_subset=True, filter_out_zuco=True, clean_txt=False):

        
        
        label_dict_inv = pickle.load(open(os.path.join(wikirel_folder, 'label_dict.p'),'rb'))
        label_dict = {v: k for k, v in label_dict_inv.items()}
        self.label_dict_inv = label_dict_inv
        
        self.train_set, self.validation_set, self.test_set = load_relations(wikirel_folder, label_dict, filter_out_zuco=filter_out_zuco, zuco_version=1, tokenizer=tokenizer)   
        
        if zuco_subset:
            # Use subset of labels!
            zuco_relations = ['award', 
                      'education',
                      'employer', 
                      'founder', 
                      'job_title', 
                      'nationality', 
                      'political_affiliation',
                      'visited',
                      'wife']
            
            # Filtering and remapping labels
            label_dict_temp = {k:self.label_dict_inv[k] for k in zuco_relations}
            self.train_set = get_subset(self.train_set,label_dict_temp)
            self.validation_set = get_subset(self.validation_set,label_dict_temp)
            self.test_set = get_subset(self.test_set,label_dict_temp)
            self.label_dict_inv = {k:i for i,k in enumerate(label_dict_temp.keys())}
            label_dict = {v:k for k, v in self.label_dict_inv.items()}

                
            print('Using only labels in Zuco relations', self.label_dict_inv)
            

        self.n_classes = len(set(self.train_set['label']))


        self.train_set.columns.add('index')
        self.test_set.columns.add('index')
        self.validation_set.columns.add('index')
        self.train_set.columns.add('weight')
        self.test_set.columns.add('weight')
        self.validation_set.columns.add('weight')

        text_corpus = [row['text'] for row in datasets_iterator(self.train_set,self.validation_set, self.test_set)]

        
        tokenize, self.encoder = get_tokenizer_and_encoder(encoder, tokenizer, self.label_dict_inv, text_corpus=text_corpus)

        self.padding_idx = int(tokenize.pad_token_id) if tokenizer in ['bert', 'roberta-base'] else 0
    
            
        for row in datasets_iterator(self.train_set,self.validation_set, self.test_set):
            row['text'], row['label'] = self.encoder.process_example((row['label'],row['text']))
            row['weight'] = tr.empty(0)

        # Get indices after pre-processing
        for i, row in enumerate(self.train_set):
            row['index'] = i

        for i, row in enumerate(self.test_set):
            row['index'] = i
            
        for i, row in enumerate(self.validation_set):
            row['index'] = i
            
       # import pdb;pdb.set_trace()
        train_labels = [label_dict[int(d['label'])] for d in self.train_set]
        train_label_distribution = collections.Counter(train_labels)
        print('train_label_distribution',  train_label_distribution)


    
def ham_dist(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Undefined")
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def search_min_dist(source,search):
    # source_string = "pGpEusuCSWEaPOJmamlFAnIBgAJGtcJaMPFTLfUfkQKXeymydQsdWCTyEFjFgbSmknAmKYFHopWceEyCSumTyAFwhrLqQXbWnXSn"
    # search_string = "tyraM"
    # search_min_dist(source_string,search_string)
    # (28, 2, 'tcJaM')
    l = len(search)
    index = 0
    min_dist = l
    min_substring = source[:l]    
    for i in range(len(source)-l+1):
        d = ham_dist(search, source[i:i+l])
        if d<min_dist:
            min_dist = d
            index = i
            min_substring = source[i:i+l]  
    return (index,min_dist,min_substring)
    

def proc_data(dset, lower):
    data = []
    index=0
    for row in dset:
        rows = {}
        rows['text'] =  row[0].lower() if lower==True else row[0]
        rows['label'] = row[1][2]
        data.append(rows)
        index+=1
    return Dataset(data)



class ZucoDetector(object):
    def __init__(self,data_path = os.path.join(zuco_files, 'datasets','ZuCo'), zuco_version=1):

        ZM = ZucoLoader(zuco1_prepr_path=zuco_files, zuco2_prepr_path=None)
        self.zuco_texts  = list(map(list,ZM.get_zuco_task_df(zuco_version=zuco_version, task='TSR').words.unique()))
        self.tzucos=[_condense_text(' '.join(t)) for t in list(self.zuco_texts)]
        self.zuco_inds = list(range(len(self.tzucos )))     
        self.filter_inds = []
        self.in_zuco_samples = []
        
        
    def filter_zuco(self, t, threshold_frac=0.4):
        from Levenshtein import distance
        # Return False if t in tzucos
        t_readable = t
        t = _condense_text(''.join(t))
        edit_dist = [distance(t, tz) for tz in self.tzucos]
        ind = np.argmin(edit_dist)
        mind = edit_dist[ind] #search_min_dist(t, tzucos[ind])[1]

        threshold = int(threshold_frac*len(t))
        # Check if t has Levenshtein distance <= threshold
        if mind <=threshold:
            if mind >0:
                print('WIKI',t)
                print('ZUCO', self.tzucos[ind], mind, ind )
                print()
            return (False, ind)

        else:
            # Check if t contains a substring of zuco
            edit_dist = [search_min_dist(t, tz)[1] for tz in self.tzucos]
            ind = np.argmin(edit_dist)
            mind = edit_dist[ind]
            if mind <=1:
                print('Substring detected')
                print('WIKI',t)
                print('ZUCO', self.tzucos[ind], mind, ind )
                print()
                return (False,ind)
            else:
                # Check if tz is substring of t
                edit_dist = [search_min_dist(tz,t)[1] for tz in self.tzucos]
                ind = np.argmin(edit_dist)
                mind = edit_dist[ind]

                if mind <=1:
                    print('Substring detected reversed')
                    print('WIKI',t)
                    print('ZUCO', self.tzucos[ind], mind, ind )
                    print()
                    return (False,ind)
                else:
                    return (True, None)


    def filter_dataset(self, dst, ):
        dst_filtered = []
        for x in dst:
            hilf =  self.filter_zuco(x[0])
            if hilf[0]==True:
                dst_filtered.append(x)
            else:
                self.filter_inds.append(hilf[1])
                self.in_zuco_samples.append(x)
        return dst_filtered
        
    def get_not_filtered(self):
        # Get texts that are no in wiki dataset (assuming these should be controls)
        not_filtered = list(set(self.zuco_inds)- set(self.filter_inds))
        texts = [self.tzucos[i] for i in not_filtered]
        return list(zip(not_filtered, texts))
    
    def get_filtered_data(self, directory, name):
        
        filtered_file = os.path.join(directory, name)
        print(filtered_file, os.path.exists(filtered_file))
        if os.path.exists(filtered_file):
            data = pickle.load(open(filtered_file,'rb'))
        
        else:
            data = pickle.load(open(os.path.join(directory, 'wiki_rel.p'),'rb'))
            data_test = data['test']
            data_train = data['train']
            data_test = self.filter_dataset(data_test)
            data_train = self.filter_dataset(data_train)
            data = {'train':data_train,
                    'test':data_test,
                    'wiki_in_zuco_samples': self.in_zuco_samples,
                    'remaining_zuco': self.get_not_filtered()} 
            
            pickle.dump(data, open(filtered_file,'wb'))
            
        self.remaining_zuco_texts = data['remaining_zuco']
        return data['train'], data['test'], data['wiki_in_zuco_samples']

    
    def get_original_zuco_texts(self):
        import pandas as pd
        text_table_dir = 'data/ZuCo/preprocessed/texts_table.p'
        zuco_texts = pd.read_pickle(text_table_dir)
        zuco_wiki_texts = zuco_texts[zuco_texts.task_id=='TSR']
        unique_texts = set(zuco_wiki_texts[zuco_wiki_texts._is_control == True].text)
        unique_texts=[_condense_text(t) for t in list(unique_texts)]        
        return unique_texts
    
    
    def test_dist_text_to_B(self,text,B):
        
        edit_dist = [search_min_dist(text, tz)[1] for tz in B]
        ind = np.argmin(edit_dist)
        mind = edit_dist[ind]
        if mind != 0:
            print('Check control match - distance nonzero!')
            print('Not in WIKI',text)
            print('ZuCo CONTROL', B[ind], mind, ind )
            print()


def load_relations(directory, label_dict, filter_out_zuco=False, zuco_version=1, tokenizer=None):

    
    lower= False if 'roberta' in tokenizer or 't5' in tokenizer else True
    
    data = pickle.load(open(os.path.join(directory, 'wiki_rel.p'),'rb'))

    data_test = data['test']
    data_train = data['train']

    
    all_text_gt = [ _condense_text(''.join(x[0])) for x in data_train] + [ _condense_text(''.join(x[0])) for x in data_test]

    zuco_detect = ZucoDetector(zuco_version=zuco_version)
    
    n_train, n_test = len(data_train), len(data_test)
    
    print('train {}, test {}'.format( n_train, n_test))

        
    if filter_out_zuco == True:

        data_train, data_test, data_in_zuco = zuco_detect.get_filtered_data(directory, name = 'wiki_rel_zuco_filtered.p')      
        undetected_zuco_texts =  zuco_detect.remaining_zuco_texts
        
        n_train_filt, n_test_filt = len(data_train), len(data_test)
        print('train {}, test {}'.format(n_train_filt, n_test_filt))
        print('total zucos texts', len(zuco_detect.tzucos))
        print('Filtered', (n_train+ n_test) - (n_train_filt+ n_test_filt))
        print('Unique filter inds', len(undetected_zuco_texts))                                 
        print('undetected_zuco_texts', len(undetected_zuco_texts))
        
        # Check that undetected zuco texts are in control
        controls_zuco = zuco_detect.get_original_zuco_texts()
        for j, zt in undetected_zuco_texts:
            zuco_detect.test_dist_text_to_B(zt, controls_zuco)
    
    else:
        print('Loading full Wiki Relations dataset.')

    
    train_s = proc_data(data_train, lower=lower)
    
    # Split test in test/val
    nid = len(data_test)
    ids = list(range(nid))

    np.random.seed(1)
    np.random.shuffle(ids)
    
    sets = {}
    # Split test set into (test, val)
    ids_test_split, ids_val_split = ids[:int(0.5*nid)], ids[int((0.5)*nid):]
    assert len(ids_test_split)+len( ids_val_split) == nid
    
    data_test_split, data_val_split = [data_test[i] for i in ids_test_split], [data_test[i] for i in ids_val_split]
    
    
    test_s = proc_data(data_test_split, lower=lower)
    validation_s =  proc_data(data_val_split, lower=lower)


    # Using filtered texts as validation set (just for model evaluation)
    # validation_s =  proc_data(data_in_zuco)

    return train_s, validation_s, test_s
