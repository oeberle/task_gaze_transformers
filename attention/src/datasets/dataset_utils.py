from sklearn.feature_extraction.text import TfidfTransformer

import torch
import numpy as np

from ..utils.text_encoders import ClfBertProcessor, BaseProcessor
from torchnlp.encoders.text import SpacyEncoder

from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer



def get_tokenizer_and_encoder(encoder, tokenizer, label_dict_inv, text_corpus=None):
    
    if tokenizer in ['sst', 'wikirel']:
        if encoder is None:
            tokenizer='spacy'
            print('encoder not set, use spacy instead')
        else:
            encoder.label2id = label_dict_inv
            encoder = encoder
            tokenize=None
    if tokenizer == 'spacy':
        append_sos=True
        tokenize = SpacyEncoder(
            text_corpus, min_occurrences=1, append_eos=True,
            reserved_tokens=['<pad>', '<unk>', '</s>', '<mask>'])
        encoder = BaseProcessor(
            tokenize, label2id=label_dict_inv, num_max_positions=256)

    if tokenizer == 'bert':
        # BERT highlights the merging of two subsequent tokens (with ##)
        assert encoder is None
        tokenize = BertTokenizer.from_pretrained('bert-large-uncased',  do_lower_case=True)
        encoder = ClfBertProcessor(tokenize, label2id=label_dict_inv, num_max_positions=256)
    
    if tokenizer == 'roberta-base':
        # RoBERTa highlights the start of a new token with a specific unicode character (in this case, \u0120, the G with a dot). 
        assert encoder is None
        tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        tokenize.vocab = tokenize.get_vocab()
        encoder = ClfBertProcessor(tokenize, label2id=label_dict_inv, num_max_positions=256, CLS = '<s>', PAD = '<pad>', EOS = '</s>')
        
    if tokenizer == 't5-base':
        assert encoder is None
        tokenize = AutoTokenizer.from_pretrained('t5-base')
        encoder = ClfBertProcessor(tokenize, label2id=label_dict_inv, num_max_positions=256, CLS = None, PAD = '<pad>', EOS = '</s>')
        
    return tokenize, encoder



def compute_tfidf_weights(train_set, test_set, vocab_size):
    """ Compute the Tf-idf weights (based on idf vector computed from train_set)."""

    transformer = TfidfTransformer()

    # fit idf vector on train set
    counts = np.zeros((len(train_set), vocab_size), dtype=np.int64)
    for i, row in enumerate(train_set):
        counts_sample = torch.bincount(row['text'])
        counts[i, :len(counts_sample)] = counts_sample.cpu().data.numpy()
    tfidf = transformer.fit_transform(counts)

    for i, row in enumerate(train_set):
        row['weight'] = torch.tensor(tfidf[i, row['text']].toarray().astype(np.float32).flatten())

    # compute tf-idf weights for test set (using idf vector from train set)
    counts = np.zeros((len(test_set), vocab_size), dtype=np.int64)
    for i, row in enumerate(test_set):
        counts_sample = torch.bincount(row['text'])
        counts[i, :len(counts_sample)] = counts_sample.cpu().data.numpy()
    tfidf = transformer.transform(counts)

    for i, row in enumerate(test_set):
        row['weight'] = torch.tensor(tfidf[i, row['text']].toarray().astype(np.float32).flatten())
