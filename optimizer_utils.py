import torch as tr
import torch.nn as nn
import torch.optim as optim
from torchtext import data
import time
import spacy
import warnings
import os
import operator
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager

import models

from typing import Optional, Tuple, List, Callable
from sklearn.metrics import f1_score
import collections


def f1_loss(y_pred:tr.Tensor, y_true:tr.Tensor, is_training=False, return_labels=True) -> tr.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
            
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # Macro seems the main method 
    #(see https://github.com/sebastianruder/NLP-progress/blob/master/english/relationship_extraction.md)
    f1= f1_score(y_true, y_pred, average='macro', labels=np.unique(y_pred))
    if return_labels:
        return f1, y_true, y_pred
    else:
        return f1


def get_f1_classwise(y_pred,y_true):
    un_labels = np.unique(y_pred)
    f1_classwise= f1_score(y_true, y_pred, average=None, labels=un_labels)
    f1_dict = {k:v for k,v in zip(un_labels, f1_classwise)}
    return f1_dict



def categorical_accuracy(y_pred, y_true):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = y_pred.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct   = max_preds.squeeze(1).eq(y_true)
    return correct.sum() / len(y_true) # tr.FloatTensor([y_true.shape[0]])

    

def max_norm(model, max_val=3, eps=1e-8):
    for name, param in model.named_parameters():
        if 'bias' not in name and 'embedding' not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = tr.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))
            operator.attrgetter(name.rsplit('.',1)[0])(model).weight.data.copy_(
                tr.nn.Parameter(param))
    return model


def train(model, iterator, optimizer, criterion, scheduler, masking_idx, eos_idx, task) -> None:
    
    model.train()
    for batch in iterator:
        optimizer.zero_grad()

        #### add another loop to go over words in sentence, mask word and change label
        if task=='LM':
            text_lm = []
            label_lm = []
            for isen in batch[1].T:
                for itoken, token in enumerate(isen):
                    if token==eos_idx:
                        break
                    else:
                        isen_tmp = tr.zeros(isen.shape)
                        isen_tmp.copy_(isen)
                        isen_tmp[itoken] = masking_idx
                        text_lm.append(isen_tmp)
                        label_lm.append(token)
            predictions = model(tr.stack(text_lm).long().T)
            loss = criterion(predictions, tr.stack(label_lm).long())

        elif 'bert' in model.model_name:
            predictions = model(batch[1].t().to(model.device[0]))
            loss = criterion(predictions, batch[2].long().to(model.device[0]))

        else:
            predictions = model(batch[1])
            loss = criterion(predictions, batch[2].long())



        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        if isinstance(model, (models.Kim2014CNN_1FS, models.Kim2014CNN_MFS)):
            max_norm(model)

    return

def train_model(model, train_iter, optimizer=None,criterion=None, scheduler=None,val_iter=None,
                n_epochs=25, save_best_model=True, verbose=True, zero_embs=None,
                task = None, masking_idx = None, eos_idx = None,
                save_each_epoch=False, stopping_rule: Optional[Callable[
                    [List[float], List[float]], bool]]=None, 
                sr_kwargs={}
               ) -> Tuple[List[float], List[float]]:
    """
    Uses as optimizer Adam, and as criterion Cross-entropy (by default).

    Args:
        n_epochs: Number of epochs to train. If `stopping_rule` is not None,
            this is the maximum number of epochs to train. 
        optimizer (opt): If None, will be set to 
            `torch.optim.Adam(model.parameters())`.
        criterion (opt): If None, will be set to `torch.nn.CrossEntropyLoss()`.
        save_best_model (bool): If True, the model with the best validation
            score will be saved to disk (this requires both `model.folder` and
            `model.model_name` to be set). Defaults to True.
        save_each_epoch (bool): If True, the state_dict of the model will be 
            saved after each epoch. The filename will end on `-ep{epoch_num}.pt`
            with epoch_num being the number of epochs trained. Thus, `ep0` will 
            be the state_dict of the model as it was passed to this function, 
            i.e. without training (by this function call).
        stopping_rule (callable): A function with signature
            `f(train_losses: List[float], 
               val_losses: Optional[List[float]]=None, 
               **sr_kwargs) -> bool`
            which returns True, if training should be stopped (even if n_epochs
            isn't reached yet). Use for early stopping. Defaults to None. 
            Please note that the outputs of the criterion (are greater values 
            better?) has to fit the stopping_rule function.
        sr_kwargs (dict): Dictionary of kwargs passed to `stopping_rule`.

    Returns:
        train_losses (list of float): for each epoch, the train performance 
            after training on the entire batch, where the performance is 
            computed by the passed criterion. 
        val_losses (list of float or None): As train_losses but on validation 
            iterator if given, otherwise None.
    """
    if optimizer is None:
        optimizer = optim.Adam(model.parameters())
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # init
    train_losses    = []
    val_losses      = [] if val_iter else None
    best_loss       = float("inf")
    best_state_dict = None
    best_epoch      = None
    summary = []

    if save_each_epoch:
        # save initial model (without training)
        tr.save(model.state_dict(), 
            _determine_state_dict_fpath(model, postfix="-ep0"))

    for epoch in range(n_epochs):
        start_time = time.time()
        train(model, train_iter, optimizer, criterion, scheduler, masking_idx, eos_idx, task)
        end_time = time.time()

        _set_embedding_weights_to_zero(model, zero_embs)

        train_loss, train_acc, train_f1, _,_ = score(model, train_iter, criterion)
        train_losses.append(train_loss)
        if val_iter:
            val_loss, val_acc, val_f1,_,_ = score(model, val_iter, criterion)
            val_losses.append(val_loss)
            if (val_loss < best_loss) and save_best_model:
                best_loss       = val_loss
                best_state_dict = deepcopy(model.state_dict())
                best_epoch      = epoch

        if verbose:
            print()
            summary_epoch = _print_epoch_summary(
                epoch_num=epoch, epoch_secs=end_time-start_time,
                train_loss=train_loss, train_acc=train_acc, train_f1=train_f1,
                val_loss=val_loss if val_iter else None,
                val_acc=val_acc if val_iter else None, 
                val_f1=val_f1 if val_iter else None, 
                best_val_so_far= val_loss==best_loss
            )
            summary.append(summary_epoch)

        if save_each_epoch:
            tr.save(model.state_dict(), 
                _determine_state_dict_fpath(model, postfix=f"-ep{epoch+1}"))

        if stopping_rule is not None:
            if stopping_rule(train_losses, val_losses, **sr_kwargs): # returns bool
                print("\nStopping criterion met.\n")
                break

    if save_best_model:
        save_dir = _determine_state_dict_fpath(model)
        print(f"\nSaving best model (epoch {best_epoch+1})",save_dir)
        tr.save(best_state_dict,save_dir)

    return train_losses, val_losses if val_iter else None, summary


def _determine_state_dict_fpath(model, postfix: str="") -> str:
    if hasattr(model, "folder") and hasattr(model, "model_name"):
        return os.path.join(model.folder, f"{model.model_name}-model{postfix}.pt")
    else:
        
        warnings.warn("specify model.folder and molde.model_name to save state_dict to")
        raise
        #  warnings.warn("model.folder or model.model_name not set. Saving "
        #              "state_dict to current working directory.")
        # return f"{model.__class__.__name__}-model{postfix}.pt"


def _set_embedding_weights_to_zero(model: nn.Module, 
                                   emb_inds: Optional[List[int]]=None) -> None:
    """
    Args:
        emb_inds (list of int (opt)): Indices of weights in embedding (and 
            attention if applicable to model) which are to be set to zero.
    """
    if emb_inds is not None:
        for i in emb_inds:
            model.embedding.weight.data[i] *= 0.0
            if hasattr(model, 'attention'):
                model.attention.weight.data[i] *= 0.0
    return


def _print_epoch_summary(
    epoch_num: int = None, epoch_secs: int = None,
    train_loss: float=None, train_acc: float=None, train_f1: float=None,
    val_loss: float=None, val_acc: float=None, val_f1: float=None, best_val_so_far: bool=False
) -> str:
    
    
    S = []
    if epoch_num is not None and epoch_secs is not None:
        ep_mins, ep_secs = epoch_time(0, epoch_secs)
        
        str_ = f"Epoch: {epoch_num+1:02} | Epoch Time: {ep_mins}m {ep_secs}s"
        S.append(str_)
        print(str_)
    if train_loss is not None and train_acc is not None:
        str_ = f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1*100:.2f}%"
        S.append(str_)
        print(str_)
    if val_loss is not None and val_acc is not None:
        flag = "  *" if best_val_so_far else ""
        str_ =f"\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}% |  Val. F1: {val_f1*100:.2f}%{flag}" 
        S.append(str_)
        print(str_)
        
    summary = '\n'.join(S)
    
    return summary

        

def score(model, iterator, criterion) -> Tuple[float,float]:
    """
    Computes loss and accuracy of model, while setting it temporarily to eval 
    mode.

    Returns:
        loss (float): weighted (by batch size) average of losses over all batches
        accuracy (float): weighted (by batch size) average of accuracies over 
            all batches
    """
    overall_loss = 0
    overall_acc  = 0
    overall_f1 = 0
    n = 0
    y_true =[]
    y_pred = []
    for batch in iterator:
        with evaluating(model):
            if 'bert' in model.model_name:
                raw_model_out = model(batch[1].t().to(model.device[0]))
            else:
                raw_model_out = model(batch[1])
            
            acc = categorical_accuracy(raw_model_out, batch[2].long().to(model.device[0])).item()
            loss = criterion(raw_model_out, batch[2].long().to(model.device[0])).item()
            f1_score, y_tr, y_pr = f1_loss(raw_model_out, batch[2].long().to(model.device[0]))
            f1_score = f1_score.item()

        n            += len(batch[2])
        overall_acc  += len(batch[2]) * acc
        overall_loss += len(batch[2]) * loss
        overall_f1 += len(batch[2]) * f1_score
        y_true+=list(map(int, y_tr))
        y_pred+=list(map(int, y_pr))

    f1_classwise = get_f1_classwise(y_pred,y_true)
    return overall_loss/n, overall_acc/n, overall_f1/n, y_true, f1_classwise



def evaluate(model, iterator, criterion):
    raise NotImplementedError("use function score instead")


def text_to_tensor(text: str, tokenizer, min_len = 4, device = 'cpu'):
    nlp = spacy.load("en_core_web_sm") #not sure if this is the same model as the one used earlier
    tokenized = [tok.text.lower() for tok in nlp.tokenizer(text)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [tokenizer.token_to_index[t] for t in tokenized]
    tensor = tr.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    return tensor


def predict_class(model, text: str, tokenizer, min_len = 4):
    tensor = text_to_tensor(text, tokenizer, min_len)
    preds = model(tensor)
    max_preds = preds.argmax(dim = 1)
    return max_preds.item()


def get_attention(model, text: str, tokenizer, min_len = 4):
    tensor = text_to_tensor(text, tokenizer, min_len)
    _ = model(tensor)
    return model.attn


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class TensorMaker:
    def __init__(self, 
                 TEXT,
                 spacymodel: str = "en", 
                 device: Optional[str] = None):
        """
        Args:
            TEXT (torchtext.data.field.Field): E.g. from 
                `data.Field(tokenize = 'spacy')`; with vocabulary already 
                built. 
            spacymodel (str): passed to `spacy.load()`
            device (str (opt)): Device to work on ("cuda" or "cpu"). If None, 
                will be determined automatically (cuda if available). Defaults 
                to None.
                
        Raises:
            AttribuiteError: If `TEXT` is missing the attribute `vocab`. 
                Solution: Build vocabulary with `TEXT.build_vocab(...)`.
        """
        self.nlp  = spacy.load(spacymodel)
        self.TEXT = TEXT
        
        if not hasattr(self.TEXT, "vocab"):
            raise AttributeError("TEXT must have a vocabulary already built.")
        
        if device is None:
            self.device = tr.device(
                'cuda' if tr.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
    
    def to_tensor(self, text: str, min_len: int = 4) -> tr.Tensor:
        """
        Args:
            text (str): A string to be converted to a tensor of vocab indices.
        Returns:
            torch.Tensor: Input as vocab indices. Shape (n_words, 1).
        """
        tokenized = [tok.text for tok in self.nlp.tokenizer(text)]
        if len(tokenized) < min_len:
            tokenized += [self.TEXT.pad_token] * (min_len - len(tokenized))
        indexed = [self.TEXT.vocab.stoi[t] for t in tokenized]
        tensor  = tr.LongTensor(indexed).to(self.device)
        tensor  = tensor.unsqueeze(1)
        return tensor


def plot_learning_curves(losses_train, losses_val, plot_best_val: bool=True):
    num_epochs = len(losses_train)
    fig, ax = plt.subplots(1, figsize=(10,5))

    if plot_best_val:
        best_val   = np.min(losses_val)
        best_epoch = np.argmin(losses_val)
        ax.hlines(best_val,   0, num_epochs, colors="gray", linestyles="dashed")
        ax.vlines(best_epoch, np.min(losses_train), np.max(losses_val), 
                  colors="gray", linestyles="dashed")
    
    ax.plot(range(num_epochs), losses_train, c="b", label="train")
    ax.plot(range(num_epochs), losses_val, c="r", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(loc="upper center")
    ax.set_title("Learning curves")


def early_stopping_up_rule(train_losses, val_losses, s: int=1, k: int=5, 
                           verbose: bool=False) -> bool:
    """
    Early Stopping UP_s rule.
    
    Args:
        train_losses: Ignored. Merely there for usability as `stopping_rule` 
            param in `train_model`.
        val_losses (list of float): Validation performances of all past epochs
            with the last element representing the most recent epoch.
        s (int): 1 or greater. Number of consecutive strips for which the UP_1
            rule has to yield "stop". (correct?) 
        k (int): strip length.
        
    Returns:
        bool: True, if training can be stopped following this rule.
        
    References:
        Montavon et al 2012: Neural Networks: Tricks of the Trade, 2nd edition,
            Springer, ISBN 978-3-642-35288-1, p.57
    """
    n_epochs = len(val_losses)
    n_strips = n_epochs // k # complete strips
    
    # only check on end-of-strip epochs, thus never stop "during" a strip
    epoch_is_end_of_strip = n_epochs % k == 0
    too_few_strips        = n_strips < s+1
    if too_few_strips or not epoch_is_end_of_strip:
        return False
    
    if s == 0:
        if verbose:
            _print_early_stopping_up_info(s, k, n_epochs, n_strips, 
                                          None, True)
        return True
    else:
        # decision for the strip at hand
        loss_at_t         = val_losses[-1]
        loss_at_t_minus_k = val_losses[-1-k]
        val_loss_worsened = loss_at_t > loss_at_t_minus_k
        
        # decision for s-1
        s_minus_1_decision = early_stopping_up_rule(
            None, val_losses[:(n_strips-1)*k], 
            s=s-1, k=k, verbose=verbose)
        
        decision = val_loss_worsened and s_minus_1_decision
        if verbose:
            _print_early_stopping_up_info(s, k, n_epochs, n_strips, 
                                          val_loss_worsened, decision)
        return decision


def _print_early_stopping_up_info(s, k, n_epochs, n_strips, val_loss_worsened, 
                                  decision) -> None:
    if s==0:
        print(f"Early Stopping UP-rule with k = {k}")
    else:
        print(f"\ts:{s:2d}, n_epochs:{n_epochs:3d}, n_strips:{n_strips:3d}, "
          f"loss_up: {str(val_loss_worsened):5s} -> {decision}")


@contextmanager
def evaluating(net):
    """
    Temporarily switch to evaluation mode.

    License:
        MIT License, Christoph Heindl, Jul 2018, 
        https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
    """
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()
            
            
            
def get_training_summary(model, test_loader, criterion, data_name,  train_loader=None):

    test_loss, test_acc, test_f1, y_true_test, f1_classwise_test = score(model, test_loader, criterion)
    y_count_distr = collections.Counter(y_true_test)
    y_count_distr = dict(sorted(y_count_distr.items(), key=lambda pair: pair[0], reverse=False))
    
    
    
    if train_loader:
        train_loss, train_acc, train_f1, y_train_test, f1_classwise_train = score(model, train_loader, criterion)
        labels=  sorted(list(set(y_train_test)))
        y_count_distr_train = collections.Counter(y_train_test)
        y_count_distr_train = dict(sorted(y_count_distr_train.items(), key=lambda pair: pair[0], reverse=False))

    else:
        labels=  sorted(list(set(y_true_test)))
     
    S = []
    S.append(data_name)
    S.append('Labels: {}'.format(str(labels)))
    S.append('\t N_test: {:.0f} |  Test Loss: {:.3f} |  Test Acc: {:.2f}%  |  Test F1: {:.2f}%'.format(len(y_true_test), test_loss, test_acc * 100, test_f1 * 100))
    S.append(str(y_count_distr))
   
    f1_classwise_test_str = {k:'{:.3f}'.format(v) for k,v in f1_classwise_test.items()}
    S.append('F1-classwise:\t' + str(f1_classwise_test_str))
  
                
                
    
    if train_loader:
        S.append('\t N_train: {:.0f} |  Train Loss: {:.3f} |  Train Acc: {:.2f}%  |  Train F1: {:.2f}%'.format(len(y_train_test), train_loss, train_acc * 100, train_f1 * 100))
        S.append(str(y_count_distr_train))
        f1_classwise_train_str = {k:'{:.3f}'.format(v) for k,v in f1_classwise_train.items()}
        S.append('F1-classwise:\t' + str(f1_classwise_train_str))

        
    summary = '\n'.join(S) + '\n\n'
    return summary
            
