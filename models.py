import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import TfidfTransformer
import lrp
import pandas as pd
from typing import Optional, Tuple, List, Any, Iterable, Dict, Callable
# from transformers import *
import transformers

class BertModel(torch.nn.Module):

    def __init__(self, output_dim: int, pretrained_weights = 'bert-large-uncased'):
        super(BertModel, self).__init__()     
        self.model  = BertForSequenceClassification.from_pretrained(pretrained_weights,
                                                                    num_labels=output_dim,
                                                                    output_hidden_states=True,
                                                                    output_attentions=True)

    def forward(self, text):
        outputs = self.model(text)
        logits, hidden_states, all_attentions  = outputs[0], outputs[1], outputs[2]
        self.hidden_states = hidden_states
        self.all_attentions = all_attentions
        return logits

    @property
    def is_explainable(self) -> bool:
        """
        Returns hard coded value, whether the forward_and_explain method is 
        usable.
        """
        return False
        
    def forward_and_explain(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Not implemented
        """
        raise NotImplementedError()
        

class RobertaModel(torch.nn.Module):
    # from transformers import RobertaTokenizer, RobertaModel
    def __init__(self, output_dim: int, pretrained_weights = 'roberta-base'):
        super(RobertaModel, self).__init__()     
        self.model  = transformers.RobertaModel.from_pretrained(pretrained_weights,
                                                                    num_labels=output_dim,
                                                                    output_hidden_states=True,
                                                                    output_attentions=True)
    def forward(self, text):
        outputs = self.model(text)
        logits, hidden_states, all_attentions  = outputs[0], outputs[2], outputs[3] #  outputs[1] is pooler output(?)
        self.hidden_states = hidden_states
        self.all_attentions = all_attentions
        return logits

    @property
    def is_explainable(self) -> bool:
        """
        Returns hard coded value, whether the forward_and_explain method is 
        usable.
        """
        return False
        
    def forward_and_explain(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Not implemented
        """
        raise NotImplementedError()
        
        
class T5Model(torch.nn.Module):
    # from transformers import RobertaTokenizer, RobertaModel
    def __init__(self, output_dim: int, pretrained_weights = 't5-base'):
        super(T5Model, self).__init__()     
        self.model  = AutoModelWithLMHead.from_pretrained("t5-base",
                                            output_hidden_states=True,
                                            output_attentions=True)
    def forward(self, text):
        outputs = self.model(text, decoder_input_ids=text)
        logits, hidden_states, all_attentions  = outputs['logits'], outputs['encoder_hidden_states'], outputs['encoder_attentions'] 
        self.hidden_states = hidden_states
        self.all_attentions = all_attentions
        return logits

    @property
    def is_explainable(self) -> bool:
        """
        Returns hard coded value, whether the forward_and_explain method is 
        usable.
        """
        return False
        
    def forward_and_explain(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Not implemented
        """
        raise NotImplementedError()
        
        


class Kim2014CNN_1FS(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, 
                 filter_size: int, output_dim: int, dropout: float, pad_idx: int):
        """
        CNN as developed by Kim 2014, but with only one filtersize for 
        convolutions.
        
        References:
            Kim 2014: https://arxiv.org/pdf/1408.5882.pdf
        """
        super(Kim2014CNN_1FS, self).__init__()
        self.a_minlenpad = MinLenPad(filter_size)
        self.b_permute   = Permute(1, 0)
        self.c_embedding = nn.Embedding(vocab_size, embedding_dim, 
                                        padding_idx=pad_idx)
        self.d_unsqueeze = Unsqueeze(1)
        self.e_conv = nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                                kernel_size = (filter_size, embedding_dim))
        
        self.f_relu = nn.ReLU(inplace=False) # inplace=True yields error in 
        
        self.g_squeeze = Squeeze(3)
        self.h_maxpool = MaxDimPool1d(2)
        self.i_squeeze = Squeeze(2)
        self.j_dropout = nn.Dropout(dropout)
        self.k_fc      = nn.Linear(1 * n_filters, output_dim)


    @property
    def embedding(self) -> nn.Module:
        """
        For same way of accessing this module as in all other CNN models
        """
        return self.c_embedding

    @property
    def is_explainable(self) -> bool:
        """
        Returns hard coded value, whether the forward_and_explain method is 
        usable.
        """
        return True
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (tensor): Tensor representing a text. Shape: (n_words, batch_size)
            seed (int (opt)): Seed to pass to torch.manual_seed before applying
                dropout.
        """
        x = self.a_minlenpad(x)
        x = self.b_permute(x)
        x = self.c_embedding(x)
        x = self.d_unsqueeze(x)
        x = self.e_conv(x)
        x = self.f_relu(x)
        x = self.g_squeeze(x)
        x = self.h_maxpool(x)
        x = self.i_squeeze(x)
        x = self.j_dropout(x)
        x = self.k_fc(x)
        return x
    
    
    def forward_and_explain(
        self, x,
        gamma: float = 0.0, 
        eps: float = 0.1,
        return_all: bool = False,
        raise_if_in_train_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: as in forward
            cv_rho, fc_rho (callable): Passed to argument `rho` of `lrp.relprop`
                when propagating through the convolutional layer (cv) and the 
                fully connected layer at the end (fc) respectively. If None, 
                nothing is done to the layer and its weights. Defaults to None.
            cv_eps, fc_eps (float): Epsilons used when propagating through the 
                convolutional layer (cv) and the fully connected layer at the 
                end (fc) respectively. Defaults to epsilons used in Arras et al
                2016 [p. 6].
            raise_if_in_train_mode (bool): If True, a ModelModeError will be
                raised if the model is in training mode (`model.training==True`)
                and thus not deterministic in its output. Defaults to True.

            return_all (bool): If True, the entire activations and relevances
                dictionaries are returned. Defaults to False.
                
        Returns:
            tuple of 2 tensors: output of model, relevances on word-level

            If return_all is True, two dictionaries are returned: 
                activations, relevances, where 
                    `activations["x"]` is x,
                    `activations["j_fc"]` is the output of the model, 
                and
                    `relevances["j_fc"]` is the output of the model masked by
                    the predicted class (only the value of the predicted class 
                    != 0).

        Raises:
            ModelModeError: see raise_if_in_train_mode in Args

        Details:
            Methodology from Arras et al 2016

        References:
            Arras et al 2016: https://arxiv.org/abs/1612.07843
        """
        if raise_if_in_train_mode and self.training:
            raise ModelModeError(
                "Model is in training mode and thus not deterministic")
        
        # args

        # get all activations and init relevances
        A = lrp.get_activations(x, self)
        R = lrp.init_relevances(A)
        M = self._modules # Dict[str, nn.Module]

        assert len(A) == 12, \
            "Something went wrong. A is of unexpected length. Code won't fit."

        # make dictionaries for less error-prone coding
        steps = lrp.steps(self) # x, a_permute, b_embedding, ..., j_fc

        # Dict which maps step -> output of this step
        A: Dict[str, torch.Tensor] = {s: a for s,a in zip(steps, A)} 

        # Dict which maps step -> relevance that needs to be propagated 
        # through this step)
        R: Dict[str, torch.Tensor] = {s: r for s,r in zip(steps, R)} 

        # START PROPAGATING RELEVANCES ----------------------------------------
        # propagate out-logit of predicted class through fc module
        R["j_dropout"] = lrp.relprop(A["j_dropout"], M["k_fc"], R["k_fc"], 
                                     gamma=0., eps=eps) 

        # dropout doesn't affect relevances, just copy over
        R["i_squeeze"] = R["j_dropout"]

        # revert the squeeze operation performed in forward by h_squeeze
        R["h_maxpool"] = R["i_squeeze"].unsqueeze(2)

        # propagate through max pool: winner takes all 
        # [c.f. Arras et al 2016, p. 6]
        max_mask = lrp._is_max_in_last_dim(A["g_squeeze"])
        #    in rare cases (probably only for toydata) there might be several 
        #    True values for one filter => make all except the first True
        #    value False
        max_mask       = lrp._uniquify_trues_in_last_dim(max_mask)
        R["g_squeeze"] = R["h_maxpool"] * max_mask

        # revert the squeeze operation performed in forward by f_squeeze
        R["f_relu"] = R["g_squeeze"].unsqueeze(3)

        # relu doesn't affect relevances, just copy over
        R["e_conv"] = R["f_relu"]

        # propagate through conv layer
        R["d_unsqueeze"] = lrp.relprop(A["d_unsqueeze"], M["e_conv"], 
                                       R["e_conv"], gamma=gamma, eps=eps) 


        # revert the unsqueeze operation performed in forward by c_unsqueeze
        R["c_embedding"] = R["d_unsqueeze"].squeeze(1)

        # Obtain word-level relevances by summing over embeding-dimension 
        # [c.f. Arras et al 2016, p. 6]
        R["b_permute"] = R["c_embedding"].sum(2)

        # revert the permute operation performed in forward by a_permute
        # to obtain relevances in x shape
        R["a_minlenpad"] = R["b_permute"].permute(1,0)

        # remove padding if was added to x in the beginning
        R["x"] = _unpad_relevances(
            R["a_minlenpad"], n_words_original = x.shape[0], 
            assert_pad_relevances_are_zero = True)
        # LRP DONE ------------------------------------------------------------

        assert R["x"].shape == x.shape, \
            "Computed relevances do not fit shape of x"

        if return_all:
            return A, R
        else:
            return A["k_fc"].data, R["x"].data


class Kim2014CNN_MFS(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, 
                 filter_sizes: List[int], output_dim: int, dropout: float, 
                 pad_idx: int):
        """
        CNN as developed by Kim 2014 (WITH multiple filter sizes (MFS)).
        
        References:
            Kim 2014: https://arxiv.org/pdf/1408.5882.pdf
        """
        super(Kim2014CNN_MFS, self).__init__()
        self.a_minlenpad = MinLenPad(max(filter_sizes))
        self.b_permute   = Permute(1, 0)
        self.c_embedding = nn.Embedding(vocab_size, embedding_dim, 
                                        padding_idx=pad_idx)
        self.d_unsqueeze = Unsqueeze(1)

        n_filtersizes = len(filter_sizes)
        self.filter_sizes = filter_sizes
        # Branched part of network
        self.e_branch = Branch(n_filtersizes)
        self.f_conv   = BranchedModules([
            nn.Conv2d(in_channels=1, out_channels=n_filters, 
                      kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        
        self.g_relu = BranchedModules([nn.ReLU(inplace=False)] * n_filtersizes)

        
        self.h_squeeze = BranchedModules([Squeeze(3)]      * n_filtersizes)
        self.i_maxpool = BranchedModules([MaxDimPool1d(2)] * n_filtersizes)
        self.j_squeeze = BranchedModules([Squeeze(2)]      * n_filtersizes)
        self.k_merge   = Merge(dim=1)
        # end of branched part of network

        self.l_dropout = nn.Dropout(dropout)
        self.m_fc      = nn.Linear(n_filtersizes * n_filters, output_dim)
        
        # LRP setup
        self.lrp_layers = (nn.Linear, nn.Conv2d) # (not nn.Embedding)
        self.maxpool_to_avgpool = (
            lambda m: AvgDimPool1d(m.dim) if isinstance(m, MaxDimPool1d) else m
        )
        self.n_filters_per_branch = n_filters
        self.n_branches = n_filtersizes


    @property
    def embedding(self) -> nn.Module:
        """
        For same way of accessing this module as in all other CNN models
        """
        return self.c_embedding
        

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (tensor): Tensor representing texts. shape (n_words, n_examples)
        """
        # if x.shape[0] < max(self.filter_sizes):
        #     m = nn.ZeroPad2d((0, 0, 0, max(self.filter_sizes) - x.shape[0]))
        #     x = m(x)
        x = self.a_minlenpad(x)
        x = self.b_permute(x)
        x = self.c_embedding(x)
        x = self.d_unsqueeze(x)
        x = self.e_branch(x)
        x = self.f_conv(x)
        x = self.g_relu(x)
        x = self.h_squeeze(x)
        x = self.i_maxpool(x)
        x = self.j_squeeze(x)
        x = self.k_merge(x)
        x = self.l_dropout(x)
        x = self.m_fc(x)
        return x


    @property
    def is_explainable(self) -> bool:
        """
        Returns hard coded value, whether the forward_and_explain method is 
        usable.
        """
        return True
    
    
    def forward_and_explain(
        self, x, 
        gamma: float = 0.0, 
        eps: float = 0.1,
        max_out_only: bool = True, gold_label = [],
        return_all: bool = False, raise_if_in_train_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: as in forward
            cv_rho, fc_rho (callable): Passed to argument `rho` of `lrp.relprop`
                when propagating through the convolutional layer (cv) and the 
                fully connected layer at the end (fc) respectively. If None, 
                nothing is done to the layer and its weights. Defaults to None.
            cv_eps, fc_eps (float): Epsilons used when propagating through the 
                convolutional layer (cv) and the fully connected layer at the 
                end (fc) respectively. Defaults to epsilons used in Arras et al
                2016 [p. 6].
            max_out_only (bool): Whether to propagate only the maximum 
                activation back through the network as usually done in LRP. 
                Defaults to True.
            raise_if_in_train_mode (bool): If True, a ModelModeError will be
                raised if the model is in training mode (`model.training==True`)
                and thus not deterministic in its output. Defaults to True.

            return_all (bool): If True, the entire activations and relevances
                dictionaries are returned. Defaults to False.
                
        Returns:
            tuple of 2 tensors: output of model, relevances on word-level

            If return_all is True, two dictionaries are returned: 
                activations, relevances, where 
                    `activations["x"]` is x,
                    `activations["m_fc"]` is the output of the model, 
                and
                    `relevances["m_fc"]` is the output of the model masked by
                    the predicted class (only the value of the predicted class 
                    != 0).

        Raises:
            ModelModeError: see raise_if_in_train_mode in Args

        Details:
            Methodology from Arras et al 2016. However, Arras et al didnt'
            implement a CNN with several kernelsizes as described by Kim 2014.
            As this is possible with this Module, relevances that are propagated
            through the convolutional modules, are summed up to form the 
            relevance tensors for the embedding layer. In more detail:
            After propagating through all `n_filtersizes` convolutional modules, 
            there are N relevance tensors, each of shape 
                `(batch_size, n_words, n_embedding_dims)`.
            These are summed which results in a single tensor of this shape.

        References:
            Arras et al 2016: https://arxiv.org/abs/1612.07843
            Kim 2014: https://arxiv.org/pdf/1408.5882.pdf
        """
        if raise_if_in_train_mode and self.training:
            raise ModelModeError(
                "Model is in training mode and thus not deterministic")


        # get all activations and init relevances
        A = lrp.get_activations(x, self)
        R = lrp.init_relevances(A, max_only=max_out_only, gold_label=gold_label)
        M = self._modules # Dict[str, nn.Module]

        assert len(A) == 14, \
            "A is of unexpected length. Code won't fit."

        # make dictionaries for less error-prone coding
        steps = lrp.steps(self) # x, a_minlenpad, b_permute, ..., m_fc

        # Dict which maps step -> output of this step
        A: Dict[str, torch.Tensor] = {s: a for s,a in zip(steps, A)} 

        # Dict which maps step -> relevance that needs to be propagated 
        # through this step)
        R: Dict[str, torch.Tensor] = {s: r for s,r in zip(steps, R)}

        # START PROPAGATING RELEVANCES ----------------------------------------
        # propagate out-logit of predicted class through fc module
        R["l_dropout"] = lrp.relprop(A["l_dropout"], M["m_fc"], R["m_fc"], 
                                     gamma=0., eps=eps) 

        # dropout doesn't affect relevances, just copy over
        R["k_merge"] = R["l_dropout"]

        # BRANCHED PART OF MODEL - - - - - - - - - - - - - - - - - - - - - - - 
        # init empty lists for branched part
        for module_name in ["j_squeeze", "i_maxpool", "h_squeeze", "g_relu", 
                            "f_conv", "e_branch"]:
            R[module_name] = [None] * self.n_branches

        # propagate relevances through all branches
        for i in range(self.n_branches):
            # revert the merge op., i.e. split what merge concatenated
            # shape(1, 3*n_filters) => [shape(1, n_filters)] * 3
            slice_start = self.n_filters_per_branch * i
            slice_end   = self.n_filters_per_branch * (i+1)
            R["j_squeeze"][i] = R["k_merge"][:, slice_start:slice_end]
            
            # revert the squeeze operation performed in forward by i_squeeze
            R["i_maxpool"][i] = R["j_squeeze"][i].unsqueeze(2)
            
            # propagate through max pool: winner takes all 
            # [c.f. Arras et al 2016, p. 6]
            max_mask = lrp._is_max_in_last_dim(A["h_squeeze"][i])
            #    in rare cases (probably only for toydata) there might be several 
            #    True values for one filter => make all except the first True
            #    value False
            max_mask          = lrp._uniquify_trues_in_last_dim(max_mask)
            R["h_squeeze"][i] = R["i_maxpool"][i] * max_mask
            
            # revert the squeeze operation performed in forward by g_squeeze
            R["g_relu"][i] = R["h_squeeze"][i].unsqueeze(3)
            
            # relu doesn't affect relevances, just copy over
            R["f_conv"][i] = R["g_relu"][i]
            
            # propagate through conv layer
            R["e_branch"][i] = lrp.relprop(
                A["e_branch"][i], M["f_conv"].modulelist[i], 
                R["f_conv"][i], gamma=gamma, eps=eps) 


        assert torch.all(torch.cat(R["j_squeeze"], dim=1) == R["k_merge"]
                        ).item(), \
            "Reverting merge step failed"
            
        # revert the branch operation, i.e. aggregate the relevances by summing them
        R["d_unsqueeze"] = R["e_branch"][0]
        for i in range(1, self.n_branches):
            R["d_unsqueeze"] += R["e_branch"][i]
        # END OF BRANCHED PART OF MODEL  - - - - - - - - - - - - - - - - - - - 
            
        # revert the unsqueeze operation performed in forward by c_unsqueeze
        R["c_embedding"] = R["d_unsqueeze"].squeeze(1)

        # Obtain word-level relevances by summing over embeding-dimension 
        # [c.f. Arras et al 2016, p. 6]
        R["b_permute"] = R["c_embedding"].sum(2)

        # revert the permute operation performed in forward by a_permute
        # to obtain relevances in x shape
        R["a_minlenpad"] = R["b_permute"].permute(1,0)

        # remove padding if was added to x in the beginning
        R["x"] = _unpad_relevances(
            R["a_minlenpad"], n_words_original = x.shape[0], 
            assert_pad_relevances_are_zero = True)

        # LRP DONE ------------------------------------------------------------

        assert R["x"].shape == x.shape, \
            "Computed relevances do not fit shape of x"

        if return_all:
            return A, R
        else:
            return A["m_fc"].data, R["x"].data

        
class WeightedBOE(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, 
                 filter_size: int, output_dim: int, dropout: float, 
                 zero_inds: List[int], pad_idx: int, n_classes: int):
        """
        Issues:
            [ ] handling masking of padding/zero-embeds (+ cuda/cpu)
            [ ] lrp (simplify forward if needed?)
            [ ] need to set attn weights for pad embedding to zero?  
        """
        super(WeightedBOE, self).__init__()
        self.a_permute = Permute(1, 0)
        self.b_embedding = nn.Embedding(vocab_size, embedding_dim, 
                                        padding_idx=pad_idx)
        self.classification_head = nn.Sequential(nn.Linear(embedding_dim, 
                                                           n_classes))
        self.zero_inds = zero_inds
        self.attention = nn.Embedding(vocab_size, 1, padding_idx=pad_idx)

    @property
    def embedding(self) -> nn.Module:
        """
        For same way of accessing this module as in all other CNN models
        """
        return self.b_embedding

    @property
    def is_explainable(self) -> bool:
        """
        Returns hard coded value, whether the forward_and_explain method is 
        usable.
        """
        return False
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (tensor): Tensor representing a text. Shape: (n_words, batch_size)
            seed (int (opt)): Seed to pass to torch.manual_seed before applying
                dropout.
        """
        x = self.a_permute(x)
        # mask==0. if it was a valid embedding
        mask = (torch.stack([torch.tensor(x==i, dtype=float).to(x.device) for i in self.zero_inds]).sum(0) == 0.)
        valid_length = mask.sum(dim=1)
        emb = self.b_embedding(x) 
        hilf = self.attention(x)
        self.attn = F.softmax(hilf, dim=1)
        x = self.attn*emb
        # compute average
        x =  x.sum(1)/mask.sum(dim=1).unsqueeze(1)
        x = self.classification_head(x)
        return x
    
    
    def forward_and_explain(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Not implemented
        """
        raise NotImplementedError()


def _unpad_relevances(r: torch.Tensor, n_words_original: int, 
                      assert_pad_relevances_are_zero: bool = True
                     ) -> torch.Tensor:
    """
    Removes the relevances of the paddings in a relevance tensor, given the 
    number of words in the original x for which the relevances at hand were 
    computed. 

    Args:
        r (tensor): Relevances as in LRP. shape (n_words, n_examples)
        n_words_original (int): number of words in the original x for which the 
            relevances at hand were computed.
        assert_pad_relevances_zero (bool): If True and if padding was added, 
            the corresponding relevances are asserted to be close to 0.
    
    Returns:
        r (tensor): Relevances of shape (n_words_original, n_examples).
    """
    n_words_padded = r.shape[0]
    if n_words_padded > n_words_original:
        r = r[ :n_words_original ]
        if assert_pad_relevances_are_zero:
            relevances_of_paddings = r[ n_words_padded: ]
            assert torch.allclose(relevances_of_paddings, torch.zeros(1)), \
                "padding got relevances != 0"
    return r



from attention.src.networks.self_attention import SelfAttention, SelfAttentionConv

class SelfAttentionModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, 
                 filter_size: int, output_dim: int, dropout: float, 
                 zero_inds: List[int], pad_idx: int, n_classes: int, 
                 n_attention_heads: int, compute_relevances:False):
        """
        Issues:

        """
        super(SelfAttentionModel, self).__init__()
        self.a_permute = Permute(1, 0)
        self.b_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.zero_inds = zero_inds
        self.n_attention_heads = n_attention_heads
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttentionConv(hidden_size=embedding_dim, attention_size=100, n_attention_heads=n_attention_heads, compute_relevances=compute_relevances)
        self.flatten = Flatten(1,-1)
        self.classification_head =  nn.Sequential(nn.Linear(embedding_dim*n_attention_heads, n_classes))#,

        
        
    @property
    def embedding(self) -> nn.Module:
        """
        For same way of accessing this module as in all other CNN models
        """
        return self.b_embedding    

    @property
    def is_explainable(self) -> bool:
        """
        Returns hard coded value, whether the forward_and_explain method is 
        usable.
        """
        return True
    
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (tensor): Tensor representing a text. Shape: (n_words, batch_size)
            seed (int (opt)): Seed to pass to torch.manual_seed before applying
                dropout.
        """
        x = self.a_permute(x)
        emb = self.b_embedding(x) 
        # emb.shape = (sentence_length, batch_size, hidden_size)
        M = self.self_attention(emb) #, padding_mask = None)
        A = self.self_attention.A
        # A.shape = (batch_size, n_attention_heads, sentence_length)
        # M.shape = (batch_size, n_attention_heads, hidden_size)
        
        x = self.flatten(M)
        #Check padding mask above!
        self.attn = A.squeeze()
        x = self.classification_head(x)
        return x
    
    
    def forward_and_explain(self, x, 
                            gamma: float = 0.0, 
                            eps: float = 0.1,
                            return_all: bool = False,
                            max_out_only: bool = True, gold_label = [],
                            raise_if_in_train_mode: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Not implemented
        """
        
        def pos_neg(x):
            p = sum(x[x>=0])
            n = sum(x[x<0])
            total = p+abs(n)
            return p/total, abs(n)/total      
        
        if raise_if_in_train_mode and self.training:
            raise ModelModeError(
                "Model is in training mode and thus not deterministic")
            
 

        A = lrp.get_activations(x, self)
      #  A2 = self.forward(x)
       
        R = lrp.init_relevances(A, max_only=max_out_only,  gold_label=gold_label)

        M = self._modules # Dict[str, nn.Module]

        # make dictionaries for less error-prone coding
        steps = lrp.steps(self) # x, a_minlenpad, b_permute, ..., m_fc

        # Dict which maps step -> output of this step
        A: Dict[str, torch.Tensor] = {s: a for s,a in zip(steps, A)} 

        # Dict which maps step -> relevance that needs to be propagated 
        # through this step)
        R: Dict[str, torch.Tensor] = {s: r for s,r in zip(steps, R)}

        #['x', 'a_permute', 'b_embedding', 'self_attention', 'flatten', 'classification_head']

        
        ##############
         # START PROPAGATING RELEVANCES ----------------------------------------
        # propagate out-logit of predicted class through fc module
        
        #relprop(a, layer, R, rho=bypass, eps=0):
        
        
#        print("classification_head", pos_neg( R["classification_head"] ),  R["classification_head"].shape)
        
        hilf =  torch.zeros_like(R["classification_head"])
        hilf[:,torch.argmax(R["classification_head"])] = 1.

        R["classification_head"] = hilf
        R["flatten"] = lrp.relprop(A["flatten"], M["classification_head"], R["classification_head"], 
                                     gamma=0., eps=eps) 

        # propagate through conv layer
        R["flatten"] = R["flatten"].view((-1, self.n_attention_heads, self.embedding_dim)).unsqueeze(0)
        
        
        R["self_attention"] = R["flatten"].view((-1, self.n_attention_heads, self.embedding_dim))#.unsqueeze(0)
        
        #lrp.convprop(A["self_attention"], M["flatten"], R["flatten"], rho=cv_rho, eps=cv_eps, debug=True)

     #   print("self_attention", pos_neg( R["self_attention"] ), R["self_attention"].shape)

        hilf = A["b_embedding"]
        sen_len =  hilf.shape[1]
        hilf = hilf.transpose(2,1).reshape(-1, 300*sen_len).unsqueeze(0).unsqueeze(0)
        

        R["b_embedding"] =  lrp.convprop(hilf, M["self_attention"].Attention, 
                                       R["self_attention"], extra=True, gamma=gamma, eps=eps) 

        
        
       # print("b_embedding", pos_neg( R["b_embedding"] ), R["b_embedding"].shape)
        
        
        # Revert hilf
        R["b_embedding"] =  R["b_embedding"].squeeze(0).squeeze(0).reshape(-1, 300, sen_len).transpose(2,1)

        
        # Obtain word-level relevances by summing over embeding-dimension 
        # [c.f. Arras et al 2016, p. 6]
        R["a_permute"] = R["b_embedding"].sum(2)

       # print("a_permute", pos_neg( R["a_permute"] ),  R["a_permute"].shape)
        #print()
        
        # revert the permute operation performed in forward by a_permute
        # to obtain relevances in x shape
        R["x"] = R["a_permute"].permute(1,0)

        # remove padding if was added to x in the beginning
      #  R["x"] = _unpad_relevances(
      #      R["a_minlenpad"], n_words_original = x.shape[0], 
      #      assert_pad_relevances_are_zero = True)
        # LRP DONE ------------------------------------------------------------
        
        assert R["x"].shape == x.shape, \
            "Computed relevances do not fit shape of x"

        if return_all:
            return A, R
        else:
            return A["classification_head"].data, R["x"].data


class WordIdCountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_ids: List[int]=[], drop_value: int=-1):
        self.drop_ids   = drop_ids
        self.drop_value = drop_value
        
    def fit(self, X, y=None):
        if self._is_fitted:
            raise Exception("This instance is already fitted. Init a new one.")
        X = self._preprocess(X)
        X = X.flatten()
        X = X[X != self.drop_value]
        self.vocab_ = np.unique(X)
        
        return self
        
    def transform(self, X, to_input_shape: bool=False) -> np.ndarray:
        check_is_fitted(self)
        
        if to_input_shape:
            X_as_passed = X # needed later
        
        X = self._preprocess(X)
        counts = []
        for row in X:
            row = row[row != self.drop_value]
            
            # append vocab to row, such that the counts returned by unique
            # will already be shaped correctly
            # (each count will be decreased by 1 later)
            row         = np.r_[row, self.vocab_]
            _, counts_i = np.unique(row, return_counts=True)
            counts.append(counts_i)

        counts = np.vstack(counts)
        
        # correct for the addition of all vocab elements to the row earlier
        counts -= 1
        
        if to_input_shape:
            counts = self.to_input_shape(X_as_passed, counts)
        return counts
        
    def _preprocess(self, X) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        X = X.copy()
        drop_mask    = np.isin(X, self.drop_ids)
        X[drop_mask] = self.drop_value
        
        if self._is_fitted:
            # drop out-of-vocab-words
            drop_mask    = ~np.isin(X, self.vocab_)
            X[drop_mask] = self.drop_value
        return X
    
    @property
    def _is_fitted(self) -> bool:
        return hasattr(self, "vocab_")
    
    def get_feature_names(self):
        return self.vocab_
    
    def to_input_shape(self, X, Z) -> np.ndarray:
        assert X.shape[0] == Z.shape[0], \
            "Inputs must have the same number of documents"
        assert Z.shape[1] == len(self.vocab_), (
            "'doc_term_matrix' must have as many columns as there are terms "
            "in 'vocab'")
        
        # Z might be a sparse matrix (output of sklearns vectorizers)
        # Make it dense
        if hasattr(Z, "toarray"):
            Z = Z.toarray()
        X = self._preprocess(X) # transposes X if self.docs_dim == 1

        W = [] # output array to be; for now list of rows of future output 
        for doc, weights in zip(X, Z): # iter over rows of X and Z
            unique_terms_in_doc = np.unique(doc)

            # init weight vector for this document
            w = np.zeros(len(doc), dtype=Z.dtype)
            for t in unique_terms_in_doc:
                if t in self.vocab_:
                    # get weight assigned to this term in this document
                    pos = np.argmax(self.vocab_ == t) # position of t in vocab
                    w_t = weights[pos]

                    # place this weight on the position(s) of the respective 
                    # term in this doc
                    w[doc == t] = w_t
                else:
                    # leave the weight for this term 0 (as initialized above)
                    pass
            W.append(w)
        W = np.vstack(W)
        return W


class WordIdTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, cv_kwargs, tfidf_kwargs={}):
        self.cv_kwargs    = cv_kwargs
        self.tfidf_kwargs = tfidf_kwargs
        
        self._cv    = WordIdCountVectorizer(**cv_kwargs)
        self._tfidf = TfidfTransformer(**tfidf_kwargs)
    
    def fit(self, X, y=None):
        # Fit Count Vectorizer
        self._cv.fit(X)

        # Fit TF-IDF
        C = self._cv.transform(X)
        self._tfidf.fit(C)
        
        self.vocab_ = self._cv.vocab_
        return self
    
    def transform(self, X, to_input_shape: bool=False) -> np.ndarray:
        C  = self._cv.transform(X)            
        T  = self._tfidf.transform(C) # tf-idf weights in shape (n_sentences, 
                                      #                          vocab_length)
        if to_input_shape:
            # tf-idf weights in shape (n_sentences, doc_length)
            T = self._cv.to_input_shape(X, T) 
        return T

    
    
class BNCFrequencies(nn.Module):
    '''
    http://www.kilgarriff.co.uk/bnc-readme.html
    '''
    
    def __init__(self, encoder):
        super(BNCFrequencies, self).__init__()
        self.bnc_path = '' # Add bnc path here (see bnc-readme.hmtl)
        self.df = pd.read_csv(os.path.join(bnc_path, 'bnc_proc.csv'))
        self.encoder = encoder
        
    def proc_df(self):
        df_bnc = pd.read_table(os.path.join(bnc_path,'all.num'), delimiter=' ')
        df_bnc = df_bnc.rename(columns = {'!!WHOLE_CORPUS': 'word',
                                          '!!ANY': 'pos-tag',
                                          '100106029': 'freq',
                                          '4124': 'n_files'})
        df_bnc['index'] = list(range(len(df_bnc)))
        df_bnc = df_bnc.set_index('index')

        df_proc = df_bnc.groupby('word').sum().reset_index()
        df_proc['p_word'] = df_proc['freq']/df_proc['freq'].sum()

        df_proc['inv_log_p_word'] = -np.log(df_proc['p_word'])
        x = np.array(df_proc.inv_log_p_word)
        x = (x-x.min())/(x.max()-x.min())
        df_proc['inv_log_p_word_norm'] = x
        df_proc = df_proc.sort_values(by='inv_log_p_word_norm', ascending=False)
        return df_proc


    def parameters(self):
        return None
            
    @property
    def is_explainable(self) -> bool:
        """
        Returns hard coded value, whether the forward_and_explain method is
        usable.
        """
        return False
    
    def get_freq(self, token, feature_column):
        words = list(self.df.word)
        token = token.lower()

        if ' ' in token:
            token = token.replace(' ','_')
        if token in words:
            #val= self.df[self.df.word==token].inv_freq_norm.sum()
            val = self.df[self.df.word==token][feature_column].values
            assert len(val)==1

        elif token in [ '<unk>', '<mask>', '</s>', '[SEP]', '[CLS]']:
            val=0.
        elif token in  ['<pad>', '[PAD]']:
            val=-1.
        else:
            print('Not in words', token)
            val=0.
        return np.array(val).squeeze()

        
    def forward(self, x):
        
        tokens = [[self.encoder.tokenizer.vocab[i] for i in x_] for x_ in x.T]

        freqs = [np.array([self.get_freq(t, 'inv_log_p_word_norm') for t in t_]) for t_ in tokens]
        probs = [np.array([self.get_freq(t, 'p_word') for t in t_]) for t_ in tokens]

        return  torch.Tensor(freqs).T,  torch.Tensor(probs).T



# -----------------------------------------------------------------------------
# helper modules (in order to make it easier to get activations for each layer)
#
class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims
        
    def forward(self, x):
        return x.permute(*self.dims)
    
    def __repr__(self):
        dims_as_str = ", ".join([str(d) for d in self.dims])
        return f"Permute({dims_as_str})"
    
class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return x.unsqueeze(self.dim)
    
    def __repr__(self):
        return f"Unsqueeze({self.dim})"

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return x.squeeze(dim=self.dim)
    
    def __repr__(self):
        if self.dim:
            return f"Squeeze({self.dim})"
        else:
            return f"Squeeze()"
    
class MaxDimPool1d(nn.Module):
    """
    Pools along an entire dimension (filter-size = diemnsion length)
    """
    def __init__(self, dim):
        super(MaxDimPool1d, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return F.max_pool1d(x, x.shape[self.dim])

    def __repr__(self):
        return f"MaxDimPool1d(dim={self.dim})"
    
class AvgDimPool1d(nn.Module):
    """
    Pools along an entire dimension (filter-size = diemnsion length)
    """
    def __init__(self, dim):
        super(AvgDimPool1d, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return F.avg_pool1d(x, x.shape[self.dim])

    def __repr__(self):
        return f"AvgDimPool1d(dim={self.dim})"

class BranchedModules(nn.Module):
    """
    Takes a list of modules and forwards for each one seperately.

    Args:
        modules (list of nn.Module): ...

    Attributes:
        modules (nn.ModuleList): Modules passed at init as ModuleList.
    """
    def __init__(self, modules: Iterable[nn.Module]):
        super(BranchedModules, self).__init__()
        self.modulelist = nn.ModuleList(modules)
        
    def forward(self, X: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args: 
            X (list of tensors): List of length `len(self.modules)` - one input 
                for each module.
        """
        return [m(x) for m, x in zip(self.modulelist, X)]

    def __repr__(self):
        return f"BranchedModules({repr(self.modulelist)})"


class Branch(nn.Module):
    """
    Takes a list of modules and forwards each one seperately.

    Args:
        n (int): Number of limbs following in parallel after the branch.
    """
    def __init__(self, n: int):
        super(Branch, self).__init__()
        self.n = n
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args: 
            x (list of tensors): Input to repeat for following parallel modules. 
                I.e. x -> [x, x, ..., x]
        """
        return [x.clone() for _ in range(self.n)]

    def __repr__(self):
        return f"Branch({self.n})"


class Merge(nn.Module):
    """
    Takes a list of modules and forwards each one seperately.

    Args:
        dim (int): Dimension to concatenate along. (passed to `torch.cat`)
    """
    def __init__(self, dim: int):
        super(Merge, self).__init__()
        self.dim = dim
        
    def forward(self, X: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            X (list of tensors): List of tensors coming from a branch.
        """
        return torch.cat(X, dim = self.dim)

    def __repr__(self):
        return f"Merge(dim={self.dim})"


class MinLenPad(nn.Module):
    """
    Pads examples with 0s if they are shorter than a certain length.
    E.g. min_len = 5, x = [[1],[2],[3]] -> [[1],[2],[3],[0],[0]]
    
    Args:
        min_len (int): minimum length of examples.
    """
    def __init__(self, min_len):
        super(MinLenPad, self).__init__()
        self.min_len = min_len
        
    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x (tensor): shape (n_words, n_examples)
        
        Returns:
            x (tensor): shape (d, n_examples), with d = max(min_len, n_words)
        """
        n_words_to_few = self.min_len - x.shape[0]
        if n_words_to_few > 0:
            padder = nn.ZeroPad2d((0, 0, 0, n_words_to_few))
            x = padder(x)
        return x
    
    def __repr__(self):
        return f"MinLenPad(min_len={self.min_len})"


class ModelModeError(Exception):
    def __init__(self, *args):
        self.args = args
        
        
        
        
class Flatten(nn.Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).
    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).
    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
    """
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
    
    def __repr__(self):
        return f"Flatten(flat={self.start_dim,self.end_dim})"

