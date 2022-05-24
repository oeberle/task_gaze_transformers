from sklearn.feature_extraction.text import TfidfTransformer

import torch
import numpy as np

from ..utils.text_encoders import ClfBertProcessor, BaseProcessor
from torchnlp.encoders.text import SpacyEncoder

from transformers import BertTokenizer




def get_tokenizer_and_encoder(encoder, tokenizer, label_dict_inv):
    
    if tokenizer == 'sst':
        if encoder is None:
            tokenizer='spacy'
            print('encoder not set, use spacy instead')
        else:
            encoder.label2id = label_dict_inv
            encoder = encoder
    if tokenizer == 'spacy':
        append_sos=True
        tokenize = SpacyEncoder(
            text_corpus, min_occurrences=1, append_eos=True,
            reserved_tokens=['<pad>', '<unk>', '</s>', '<mask>'])
        self.encoder = BaseProcessor(
            tokenize, label2id=label_dict_inv, num_max_positions=256)

    if tokenizer == 'bert':
        assert encoder is None

        tokenize = BertTokenizer.from_pretrained('bert-large-uncased',  do_lower_case=True)
        encoder = ClfBertProcessor(tokenize, label2id=label_dict_inv, num_max_positions=256)

    
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
