from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_EOS_INDEX, DEFAULT_UNKNOWN_INDEX
from typing import List, Tuple
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

import torch
import logging

# BertTokenizer reserved tokens: "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"

class MyBertTokenizer(BertTokenizer):
    """ Patch of pytorch_pretrained_bert.BertTokenizer to fit torchnlp TextEncoder() interface. """

    def __init__(self, vocab_file, do_lower_case=True, max_len=None, never_split=('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]'),
):
        super().__init__(vocab_file, do_lower_case=do_lower_case)
        self.append_eos = append_eos

        self.itos = list(self.vocab.keys())
        self.stoi = {token: index for index, token in enumerate(self.itos)}

        self.vocab = self.itos
        self.vocab_size = len(self.vocab)

    def encode(self, text, eos_index=DEFAULT_EOS_INDEX, unknown_index=DEFAULT_UNKNOWN_INDEX):
        """ Returns a :class:`torch.LongTensor` encoding of the `text`. """
        text = self.tokenize(text)
        unknown_index = self.stoi['[UNK]']  # overwrite unknown_index according to BertTokenizer vocab
        vector = [self.stoi.get(token, unknown_index) for token in text]
        if self.append_eos:
            vector.append(eos_index)
        return torch.LongTensor(vector)

    def decode(self, tensor):
        """ Given a :class:`torch.Tensor`, returns a :class:`str` representing the decoded text.
        Note that, depending on the tokenization method, the decoded version is not guaranteed to be
        the original text.
        """
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)
    
    
class BaseProcessor(object):
    """Base class for all neural networks."""
    CLS = '<eos>'
    PAD = '<pad>'
    

    def __init__(self, tokenizer, label2id: dict, num_max_positions:int=512):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer=tokenizer
        self.label2id = label2id
        self.num_labels = len(label2id)
        self.num_max_positions = num_max_positions
        self.vocab_size = tokenizer.vocab_size

        
        
    def process_example(self, example: Tuple[str, str]):
        "Convert text (example[0]) to sequence of IDs and label (example[1] to integer"
        assert len(example) == 2
        label, text = example[0], example[1]
        assert isinstance(text, str)
        ids = self.tokenizer.encode(text)     
        try:
            return ids, torch.tensor(self.label2id[label])
        except:
            import pdb;pdb.set_trace()

    
    
    
class ClfBertProcessor(BertTokenizer):
    
    # special tokens for classification and padding

    def __init__(self, tokenizer, label2id: dict, num_max_positions:int=512, CLS = '[CLS]', PAD = '[PAD]', EOS = '[SEP]'):
        self.tokenizer=tokenizer
        self.label2id = label2id
        self.num_labels = len(label2id)
        self.num_max_positions = num_max_positions
        self.CLS = CLS
        self.PAD = PAD
        self.EOS = EOS

    
    def process_example(self, example: Tuple[str, str]):

        "Convert text (example[0]) to sequence of IDs and label (example[1] to integer"
        assert len(example) == 2
        label, text = example[0], example[1]
        assert isinstance(text, str)
        tokens = self.tokenizer.tokenize(text)
        
        # truncate if too long 
        if len(tokens) >= self.num_max_positions:
            tokens = tokens[:self.num_max_positions-2] 
            CLS = [self.tokenizer.vocab[self.CLS]] if self.CLS else []
            ids =CLS + self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.vocab[self.EOS]]
        # pad if too short
        else:
            CLS = [self.tokenizer.vocab[self.CLS]] if self.CLS else []
            ids = CLS + self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.vocab[self.EOS]]# + pad
              
        ids = torch.LongTensor(ids)
        return ids, torch.tensor(self.label2id[label])
    
