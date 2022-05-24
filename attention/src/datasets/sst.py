import os
from ..base.torchnlp_dataset import TorchnlpDataset
from torchnlp.datasets.dataset import Dataset
from itertools import chain as datasets_iterator
from ..utils.misc import clean_text
import torch
from torchtext import data, datasets
from general_utils import _condense_text
from dataloader.data_loader_zuco import ZucoLoader
from .dataset_utils import get_tokenizer_and_encoder, compute_tfidf_weights
from _local_options import sst_folder, zuco_files

class SSTZuCo(datasets.SST):
    def __init__(self, path, text_field, label_field, zuco_texts, subtrees=False,
                 fine_grained=False, keep_only_zuco=False, **kwargs):
        """Create an SST dataset instance given a path and fields.

        Arguments:
            path: Path to the data file
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            subtrees: Whether to include sentiment-tagged subphrases
                in addition to complete examples. Default: False.
            fine_grained: Whether to use 5-class instead of 3-class
                labeling. Default: False.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]
        path = os.path.join(sst_folder, path.split('/')[-1])

        
        def get_label_str(label):
            pre = 'very ' if fine_grained else ''
            return {'0': pre + 'negative', '1': 'negative', '2': 'neutral',
                    '3': 'positive', '4': pre + 'positive', None: None}[label]
        
        def filter_zuco(t, tzucos, keep_only_zuco=False):
            t_readable = t.text
            t = _condense_text(''.join(t.text))
          #  from Levenshtein import distance
          #  edit_dist = [distance(t, tz) for tz in tzucos]
          #  mind = np.min(edit_dist)
            if keep_only_zuco:
                if t in tzucos:
                    self.filtered += [(0, t_readable) ]
                    self.n_filtered +=1
                    return True
                else:
                    return False
            else:
                if t in tzucos:
                    self.filtered += [(0, t_readable) ]
                    self.n_filtered +=1
                    return False
                else:
                    return True
            
        tzucos_cond=[_condense_text(' '.join(t)) for t in list(zuco_texts)]

        label_field.preprocessing = data.Pipeline(get_label_str)
        
        self.n_filtered = 0
        self.filtered = []
        with open(os.path.expanduser(path)) as f:
            if subtrees:
                examples = [ex for line in f for ex in
                            data.Example.fromtree(line, fields, True) ]
            else:
                examples = [data.Example.fromtree(line, fields)
                            for line in f if filter_zuco(data.Example.fromtree(line, fields),
                                                         tzucos_cond, keep_only_zuco=keep_only_zuco)]
                #examples = [data.Example.fromtree(line, fields) for line in f ]

        print('Filtered samples:', self.n_filtered, path)
        super(datasets.SST, self).__init__(examples, fields, **kwargs)


label_dict = {0: 'negative',1: 'neutral', 2: 'positive', None: None}

label_dict_inv = {v:k for k,v in label_dict.items()}

class SST_Dataset(TorchnlpDataset):

    def __init__(self, root: str, tokenizer='spacy', use_tfidf_weights=False, 
                 append_sos=False, append_eos=False, clean_txt=False, 
                 filter_out_zuco=False, train=True, test=True, keep_only_zuco=False, 
                 encoder=None, compute_tokenize_dict=False, zuco_version=1):
        super().__init__(root)

        # classes = list(range(6)) do we need this?

        # Load SST dataset

        self.train_set, self.validation_set, self.test_set = sst_dataset(
            directory=self.root, train=True, validation=True, test=True, 
            clean_txt=clean_txt, zuco_version=zuco_version, filter_out_zuco=filter_out_zuco, keep_only_zuco=keep_only_zuco, tokenizer=tokenizer)

        self.n_classes = len(set(self.train_set['label']))

        # Pre-process
        self.train_set.columns.add('index')
        self.validation_set.columns.add('index')
        self.test_set.columns.add('index')
        self.train_set.columns.add('weight')
        self.validation_set.columns.add('weight')
        self.test_set.columns.add('weight')


        # Subset train_set to normal class
        #self.train_set = Subset(self.train_set, train_idx_normal)

        # Make corpus and set encoder
        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, 
                                                                self.validation_set, 
                                                                self.test_set)]

        tokenize, self.encoder = get_tokenizer_and_encoder(encoder, tokenizer, label_dict_inv, text_corpus=text_corpus)
            
            
        self.padding_idx = int(tokenize.pad_token_id) if tokenizer in ['bert', 'roberta-base'] else 0
    
            
        # Encode
        compute_tokenize_dict = False
        k=0
        self.tokenize_dict = {}
        for row in datasets_iterator(self.train_set, self.validation_set, 
                                     self.test_set):
            text = row['text']
            row['text'], row['label'] = self.encoder.process_example(
                (row['label'], row['text']))
            
            if compute_tokenize_dict==True:
                
                ids2txt =self.encoder.tokenizer.decode(row['text']) if  tokenizer == 'bert' else  [self.encoder.tokenizer.vocab[i] for i in row['text']]
                
                self.tokenize_dict[k] = (text, row['text'], ids2txt)             
                k+=1
            
        #   if append_sos:
        #      sos_id = self.encoder.stoi[DEFAULT_SOS_TOKEN]
        #      row['text'] = torch.cat((torch.tensor(sos_id).unsqueeze(0), 
        #                               self.encoder.encode(row['text'])))


        # Compute tf-idf weights
        if use_tfidf_weights:
            compute_tfidf_weights(
                self.train_set, self.validation_set, self.test_set, 
                vocab_size=self.encoder.vocab_size)
        else:
            for row in datasets_iterator(self.train_set, self.validation_set, 
                                         self.test_set):
                row['weight'] = torch.empty(0)

        # Get indices after pre-processing
        for i, row in enumerate(self.train_set):
            row['index'] = i
        for i, row in enumerate(self.validation_set):
            row['index'] = i
        for i, row in enumerate(self.test_set):
            row['index'] = i
       
    
    # not sure if needed...
    #    _ = self.create_word_map(idx_start=1 if tokenizer=='bert' else 0)
            

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




def sst_dataset(directory='../data', train=False, validation=False, test=False, 
                clean_txt=False, filter_out_zuco=False, keep_only_zuco=False, zuco_version=1, tokenizer=None):
    
    
    
    inputs = data.Field(lower=False if 'roberta' in tokenizer or 't5' in tokenizer else 'preserve-case')
    answers = data.Field(sequential=False, unk_token=None)

    # build with subtrees so inputs are right
    if filter_out_zuco == True:
        print('Loading only Zuco' if keep_only_zuco else 'Loading SST without Zuco')
        
        ZM = ZucoLoader(zuco1_prepr_path=zuco_files, zuco2_prepr_path=None)
        zuco_texts  = list(map(list,ZM.get_zuco_task_df(zuco_version=zuco_version, task='SR').words.unique()))
        
        train_s, dev_s, test_s = SSTZuCo.splits(
                inputs, answers, zuco_texts=zuco_texts, fine_grained=False,
                train_subtrees=False, keep_only_zuco=keep_only_zuco)

    else:
        print('Loading full SST dataset.')
        train_s, dev_s, test_s = datasets.SST.splits(
            inputs, answers, root = directory, fine_grained = False,
            train_subtrees = False)
            # filter_pred=lambda ex: ex.label != 'neutral')

    ret = []
    splits = [split_set for (requested, split_set)
              in [(train,'train'), (validation,'validation'), (test,'test')] 
              if requested]

    for split_set in splits:

        if split_set=='train':
            dataset=train_s
        elif split_set=='validation':
            dataset=dev_s
        elif split_set=='test':
            dataset=test_s

        examples = []
        for id in range(len(dataset)):
            
            if clean_txt:

                text = clean_text(' '.join(dataset[id].text))
            else:
                text = ' '.join(dataset[id].text)
            label = dataset[id].label

            if text:
                examples.append({
                    'text': text,
                    'raw': text,
                    'label': label
                })

          #  import pdb;pdb.set_trace()

        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


