from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader
# from torchnlp.samplers.sorted_sampler import SortedSampler
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
import torch

class TorchnlpDataset(BaseADDataset):
    """TorchnlpDataset class for datasets already implemented in torchnlp.datasets."""

    def __init__(self, root: str, padding_idx=0):
        super().__init__(root)
        self.encoder = None  # encoder of class Encoder() from torchnlp
        self.padding_idx= padding_idx
        
    def loaders(self, batch_size: int, shuffle_train=False, shuffle_test=False, num_workers: int = 0, collate_fn=None) -> (
            DataLoader, DataLoader):
     
    
    
    
        if collate_fn is None:
           # collate_fn =  partial(_collate_fn_idx, padding_idx=self.padding_idx) 
           # collate_fn  = lambda b, params=self.padding_idx: my_collator_with_param(b, params)
        
            collate_fn = PadCollator(self.padding_idx)
        
        else:
            import pdb;pdb.set_trace()

        # Use BucketSampler for sampling
        if True:
            if hasattr(self, 'train_set'):
                train_sampler = SortedSampler(self.train_set, sort_key=lambda r: len(r['text']))
                train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, sampler=train_sampler,
                                          collate_fn=collate_fn,
                                          num_workers=num_workers)
            else:
                train_loader = None

            if hasattr(self, 'validation_set'):
                validation_sampler = SortedSampler(self.validation_set, sort_key=lambda r: len(r['text']))
                val_loader = DataLoader(dataset=self.validation_set, batch_size=batch_size, sampler=validation_sampler,
                                        collate_fn=collate_fn,
                                        num_workers=num_workers)
            else:
                val_loader = None

            if hasattr(self, 'test_set'):
                test_sampler = SortedSampler(self.test_set, sort_key=lambda r: len(r['text']))
                test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, sampler=test_sampler,
                                         collate_fn=collate_fn,
                                         num_workers=num_workers)
            else:
                test_loader = None
        else:
            print('Disabled BucketBatchSampler')
            train_sampler = None
            test_sampler = None
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, sampler=train_sampler,
                                      collate_fn=collate_fn, num_workers=num_workers)
            val_loader = DataLoader(dataset=self.validation_set, batch_size=batch_size, batch_sampler=train_sampler,
                                    collate_fn=collate_fn,
                                    num_workers=num_workers)
            test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, batch_sampler=test_sampler,
                                     collate_fn=collate_fn,
                                     num_workers=num_workers)
        return train_loader, val_loader, test_loader


class PadCollator(object):
    '''
    Allows to pass params to collate_fn
    '''
    def __init__(self, padding_index):
        self.padding_index = padding_index
    def __call__(self, batch):
        return self._collate_fn_idx(batch, self.padding_index)

    def _collate_fn_idx(self, batch, padding_index):
        """ list of tensors to a batch tensors """
        # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
        transpose = (lambda b: b.t_().squeeze(0).contiguous())

        indices = [row['index'] for row in batch]
        text_batch, _ = stack_and_pad_tensors([row['text'] for row in batch],  padding_index=padding_index)
        label_batch = torch.stack([row['label'] for row in batch])
        weights = [row['weight'] for row in batch]
        # check if weights are empty
        if weights[0].nelement() == 0:
            weight_batch = torch.empty(0)
        else:
            weight_batch, _ = stack_and_pad_tensors([row['weight'] for row in batch], padding_index=padding_index)
            weight_batch = transpose(weight_batch)

        return indices, transpose(text_batch), label_batch.float(), weight_batch


def _collate_fn_idx(batch, padding_index):
    """ list of tensors to a batch tensors """
    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t_().squeeze(0).contiguous())

    indices = [row['index'] for row in batch]
    text_batch, _ = stack_and_pad_tensors([row['text'] for row in batch],  padding_index=padding_index)
    label_batch = torch.stack([row['label'] for row in batch])
    weights = [row['weight'] for row in batch]
    # check if weights are empty
    if weights[0].nelement() == 0:
        weight_batch = torch.empty(0)
    else:
        weight_batch, _ = stack_and_pad_tensors([row['weight'] for row in batch], padding_index=padding_index)
        weight_batch = transpose(weight_batch)

    return indices, transpose(text_batch), label_batch.float(), weight_batch



def collate_fn_attns(batch):
    """ list of tensors to a batch tensors """
    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    transpose = (lambda b: b.t_().squeeze(0).contiguous())

    indices = [row['index'] for row in batch]
    text_batch, _ = stack_and_pad_tensors([row['text'] for row in batch])
    attns_batch,_ = stack_and_pad_tensors([torch.from_numpy(row['x']) for row in batch],padding_index=-1) 
    
    label_batch = torch.stack([row['label'] for row in batch])
    weights = [row['weight'] for row in batch]
    # check if weights are empty
    if weights[0].nelement() == 0:
        weight_batch = torch.empty(0)
    else:
        weight_batch, _ = stack_and_pad_tensors([row['weight'] for row in batch])
        weight_batch = transpose(weight_batch)
    return indices, transpose(text_batch), transpose(attns_batch), label_batch.float(), weight_batch

