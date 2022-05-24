from typing import List, Tuple, Callable, Union, Optional

import torch
import torch.nn as nn
import copy


# leave up here, default arg in explain()
def bypass(x, *args, **kwargs):
    """
    Does nothing. Intended for use where a callable needs to be defined.
    """
    return x


def output_computable_by_iter_through_modules(model: nn.Module, x: torch.Tensor
                                             ) -> bool:
    """
    Checks if the function `get_activations` works for a given model.
    `get_activations` only works for models for which all transformations on x
    are defined by the modules and not by functionals in their `forward` method.

    Only works for models that which don't have stochastic output in eval mode. 
    (see Details)

    Args:
        model (torch.nn.Module): Model to test.
        seed: Seed to use right before iterating through all layers to get 
            layer-wise activations with `get_activations` (will be applied via
            torch.manual_seed) and also used right before calling `model(x)`. If
            None, a seed to use this way will be drawn from the current rng 
            state.
    Return:
        bool: True, if the out-activations computed by `get_activations` are 
            close (torch.allclose) to those returned by `model(x)`. False 
            otherwise.

    Details:
        If the model is in training mode, i.e. `model.training==True`, it will
        be set to eval mode with `model.eval()` and be set back again afterwards
        by `model.train()`.
    """
    model_was_set_to_training = model.training
    model.eval()
    
    activations  = get_activations(x, model)
    model_output = model(x)

    if model_was_set_to_training:
        model.train()
    return torch.allclose(model_output, activations[-1])
    

def _explain(x: torch.Tensor, model: nn.Module, 
            considered_module_types: Tuple[nn.Module], 
            module_substitutor: Callable = bypass, seed=None
           ) -> List[torch.Tensor]:
    """
    Args:
        x: as in forward
        model: instance of nn.Module
        considered_module_types: Only modules that are of one these types
            are passed to the LRP-algorithm, all others are skipped. 
        module_substitutor: A function with signature
            `f(m: nn.Module) -> nn.Module`,
            e.g. MaxPool2d -> AvgPool2d
        seed: Seed to use right before iterating through all layers to get 
            layer-wise activations (will be applied via torch.manual_seed). 
            Passed to `get_activations`. Defaults to None.

    Returns:
        list of tensors: List of relevances, where the first element is the 
        relevance vector for the input, and the last element is the output 
        of the model.
    """
    activations = get_activations(x, model, seed=seed)
    
    modules = get_modules(model)
    n_modules = len(modules)

    relevances = init_relevances(activations)

    # loop over module outputs
    for l in range(1, n_modules)[::-1]: # reversed [l, l-1, ..., 1]
        a = activations[l]
        if isinstance(a, torch.Tensor):
            # the usual way
            pass
        elif isinstance(a, list) and isinstance(a[0], torch.Tensor):
            # branched modules
            pass

        m = module_substitutor(modules[l])

        if isinstance(m, considered_module_types):
            relevances[l] = relprop(a, m, relevances[l+1])
        else:
            relevances[l] = relevances[l+1]


    return relevances


def explain(x: torch.Tensor, model: nn.Module, 
            considered_module_types: Tuple[nn.Module], 
            module_substitutor: Callable = bypass, seed=None
           ) -> List[torch.Tensor]:
    """
    Args:
        x: as in forward
        model: instance of nn.Module
        considered_module_types: Only modules that are of one these types
            are passed to the LRP-algorithm, all others are skipped. 
        module_substitutor: A function with signature
            `f(m: nn.Module) -> nn.Module`,
            e.g. MaxPool2d -> AvgPool2d
        seed: Seed to use right before iterating through all layers to get 
            layer-wise activations (will be applied via torch.manual_seed). 
            Passed to `get_activations`. Defaults to None.

    Returns:
        list of tensors: List of relevances, where the first element is the 
        relevance vector for the input, and the last element is the output 
        of the model.
    """

    activations = get_activations(x, model, seed=seed)
    
    modules = get_modules(model)
    n_modules = len(modules)

    relevances = init_relevances(activations)

    # loop over module outputs
    for l in range(1, n_modules)[::-1]: # reversed [l, l-1, ..., 1]
        a = activations[l]
        m = module_substitutor(modules[l])

        if isinstance(m, considered_module_types):
            relevances[l] = relprop(a, m, relevances[l+1])
        else:
            relevances[l] = relevances[l+1]
            
    return relevances


def relprop(a, layer, R, gamma=0., eps=0, debug=False, extra=False):
    """
    (Mostly) from overview [Montavon et al 2019, p. 198]
    
    Args:
        a (torch.Tensor): Activations.
        rho (callable): A function that transforms the weights of a layer,
            e.g. in order to pronounce the positive weights. E.g. the LRP-gamma 
            rule would be defined via this function. See overview paper or LRP 
            tutorial. Defaults to bypass function. Together with epsilon=0, this
            results in the basic "LRP-0" rule [Montavon et al 2019, p. 196].
        layer (torch.nn.Module): ...
        eps (float): Epsilon as in LRP-epsilon rule.

    """
    if debug==True:
        import pdb;pdb.set_trace()

    if isinstance(layer, torch.nn.Sequential):
        assert len(layer)==1
        layer = layer[0]


    rho = newlayer(layer, lambda p: p + gamma*p.clamp(min=0.))
        
    a = a.data.requires_grad_(True) 
    z = eps + rho.forward(a) 

    if extra==True:
        z = z.squeeze(0).transpose(1,0)
    assert R.shape == z.shape
    s = R/(z + 1e-9)
    (z * s.data).sum().backward()
    c = a.grad
    R = a*c
    return R


def newlayer(layer,g):
    layer = copy.deepcopy(layer)
    try: 
        weights = g(layer.weight)
        layer.weight = nn.Parameter(weights)
    except AttributeError: pass

    try: layer.bias   = nn.Parameter(g(layer.bias))
    except AttributeError: pass

    return layer


def convprop(x,layer,r, gamma = 0.5, eps=0, extra=False):
    x = x.data.requires_grad_(True)
    rho = newlayer(layer,lambda p: p + gamma*p.clamp(min=0.))
    
    z = rho.forward(x)
    
    if extra==True:
        z = z.squeeze(0).transpose(1,0) 
        
    ea = ((z.clamp(min=0)**2).mean()**.5).data
    z = z + eps*ea + 1e-9
    assert r.shape == z.shape 
    (z*(r/z).data).sum().backward()
    r = (x*x.grad).data
    return r
    

def get_modules(model) -> List[nn.Module]:
    modules = list(model.children())
    return modules


def get_activations(x, model, seed = None) -> List[torch.Tensor]:
    """
    Args:
        x (tensor): input to model

    Returns:
        list of tensor: First element is x, afterwards for each module in the 
            model one activation tensor (the outputs of the respective module).
    """
    modules = get_modules(model)
    n_modules = len(modules)
    
    model.zero_grad()
    activations = [x] + [None]*n_modules
    if seed:
        torch.manual_seed(seed) # e.g. for dropout
    for l in range(n_modules):
        activations[l+1] = modules[l].forward(activations[l])
    return activations


def init_relevances(activations: List[torch.Tensor], max_only: bool=True, gold_label= []
                   ) -> List[Union[None, torch.Tensor]]:
    """
    Args:
        activations: layer wise activations, with first activation being x and
            last activation being the output of shape `(batch_size, n_classes)`.
    """
    relevances = [None] * len(activations)
    logits     = activations[-1].data
    if gold_label is not None:
        # mask logits, such that only the max one is kept, all others 0        
        label_mat = torch.zeros_like(logits)
        label_mat[:, int(gold_label)] = 1.
        relevances[-1] = label_mat * logits
    elif max_only:
        # mask logits, such that only the max one is kept, all others 0
        relevances[-1] = _is_max_in_row(logits) * logits
    else:
        relevances[-1] = logits
    return relevances


def steps(model: nn.Module) -> List[str]:
    """
    Returns ["x", {name of 1st module}, ..., {name of last module}], as these
    names are found in `model._modules`.
    """
    return ["x"] + list(model._modules.keys())


def _is_max_in_last_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Finds the max value along the last dimension and outputs a max-mask, i.e.
    a tensor of the shape of the input tensor with those elements True, 
    which are max in the last dimension, and all others False.

    For two dimensions, i.e. rows and columns: For each row, the cell with 
    the largest value is filled with a 1, all others with 0.

    E.g.
        [[2, 1, 5],        [[0, 0, 1],
         [0, 7, 3],   =>    [0, 1, 0],
         [8, 8, 9]]         [0, 0, 1]]

    Args:
        x (torch.Tensor): Tensor
    """
    n_dims      = len(x.shape)
    max_vals, _ = x.max(dim=n_dims-1, keepdim=True)
    max_mask    = torch.isclose(x, max_vals)
    return max_mask


def _is_max_in_row(a: torch.Tensor) -> torch.Tensor:
    """
    This only works on tensors with 2 dimensions! For each row, the cell with 
    the largest value is filled with a 1, all others with 0.

    E.g.
        [[2, 1, 5],        [[0, 0, 1],
         [0, 7, 3],   =>    [0, 1, 0],
         [8, 8, 9]]         [0, 0, 1]]

    Args:
        a (torch.Tensor): Tensor with two dimensions (rows and columns only), 
            e.g. of shape `(batch_size, n_classes)`.
    """
    a = a.data
    a[:,0] += 0.00000001

    argmaxs = a.data.argmax(dim=1)
    n_dims  = len(a.shape)
    assert n_dims == 2, \
        "Unexpected number of dimensions. Must be rows and columns only"
    
    mask = torch.zeros_like(a)
    mask[range(a.shape[0]), argmaxs] = 1.0
    return mask


def _uniquify_trues_in_last_dim(t) -> torch.Tensor:
    """
    Takes a bool tensor and returns one of the same shape, but with unique True
    values along the last dimension. Therefore the first Trues are taken.
    
    E.g.
    [[True,  True, False],   =>   [[True,  False, False],
     [False, True,  True]]   =>    [False,  True, False]]
    
    Args:
        t (tensor): A bool tensor of any shape.
     
    Details:
        Bools can be interpreted as 1s and 0s. Thus, adding different values to
        these bools makes all of them unique. E.g., adding [.3,.2,.1] to 
        [True,True,True] yields [1.3, 1.2, 1.1]. Computing the max-mask 
        afterwards yields [True, False, False]. 
        Added values must be smaller 1 (e.g. True-0.1 = 0.9), such that all 
        False+add_i < True+add_j, where add_i and add_j are any of the added
        values (in the example above 0.1, 0.2, and 0.3).
    """
    add = torch.linspace(True-0.1, 0, steps=t.shape[-1])
    t   = t + add
    return _is_max_in_last_dim(t)
