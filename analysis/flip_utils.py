import yaml
# from attention.src.datasets.main import load_dataset
from optimizer_utils import *
from proc.compare import get_join_inds, remove_include_toks
from general_utils import load_model_from_cfg, load_state_dict


def load_trained_model(config_file, device):
    
    with open(os.path.join('configs', config_file), 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    
    model_folder,_ = os.path.split(config_file)
    
    data_path = cfg['data_path']
    tokenizer = cfg['tokenizer']
    clean_txt = cfg['clean_txt']
    batch_size = cfg['batch_size']
    task = cfg['task']

    #set up model name for file loading and storing 
    training_dataset_name = cfg['training_dataset_name']
    model_name = cfg['model_name'] + '_' + training_dataset_name
    mode = cfg['mode'] + '_' + task

    if 'cnn' in model_name or 'self_attention' in model_name:
        model_name = model_name + '_' + mode
    elif 'bert' in model_name:
        model_name = model_name + '_' + cfg['pretrained_model'] 
        model_name = model_name + '_finetuned' if cfg['finetune_case']=='finetune' else model_name + '_untuned'
    model_folder = os.path.join(cfg['model_folder'], model_name)

    #load unfiltered base dataset to initialize tokenizer
    if task in ['SR']:
        dataset_base = load_dataset('sst', data_path, tokenizer, clean_txt=clean_txt)
        base_tokenizer= 'bert' if tokenizer=='bert' else 'sst'
        base_encoder = None if tokenizer=='bert' else dataset_base.encoder

    elif task in ['TSR','NR']:
        dataset_base = load_dataset('wiki_rel', data_path, tokenizer, clean_txt=clean_txt)
       # base_tokenizer = None # use provided encoder
        base_tokenizer = 'bert' if tokenizer=='bert' else 'wikirel'
        base_encoder = None if tokenizer=='bert' else dataset_base.encoder
        compute_f1 = True
        
    else:
        raise ValueError('specify task')

    training_dataset = load_dataset(training_dataset_name, data_path, tokenizer=base_tokenizer, clean_txt=clean_txt,
                                    encoder=base_encoder, task=task)

    # Check that training_dataset is trained on less samples (zuco samples are removed) 
    assert len(training_dataset.train_set) < len(dataset_base.train_set)
    # Check base and training use the same encoder
    assert dataset_base.encoder.tokenizer.vocab == training_dataset.encoder.tokenizer.vocab


    train_loader, val_loader, test_loader = training_dataset.loaders(batch_size, shuffle_train=True, shuffle_test=False)
    #load pretrained embeddings
    
    
    LABELS = tr.stack(train_loader.dataset['label']).unique()
    OUTPUT_DIM = len(LABELS)
    
    # Load model architecture and load external pretrained embeddings
    model, pretrained_embeddings = load_model_from_cfg(cfg, model_name, OUTPUT_DIM ,training_dataset, dataset_base, device)
    model.folder = model_folder
    load_state_dict(model, cfg, model_name, model_folder)
    print('Initialized {} from {}'.format(model_name, os.path.join(model_folder,'{}-model.pt'.format(model_name))))
    
    return model


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def flip(model, df_temp, df_ref, x_temp_ref, fracs, flip_case, UNK_IDX = 1, random_order = False,  extra_mapping={}, allowed_matches={}, remove_list=[], include_list=[], EOS = '</s>', ref_model_name=None, device='cpu', ignore_filt=[0,None]):
    """Performs the input reduction experiment for one sample"""
    
    tokens_raw = df_ref['_tokens_raw'] 
    tokens_zuco = list(df_ref['words'])
    
    if EOS.lower() not in tokens_zuco[-1].lower():
        # Should only happen for the ones that were not matched
        tokens_zuco[-1] = tokens_zuco[-1]+EOS
       
    t =  [w for w in tokens_raw if w not in remove_list]
    tokens_raw, _ =  remove_include_toks(t, include_list)
    tokens_zuco_map, tokens_spacy_map, inds_map = get_join_inds(tokens_raw, tokens_zuco, extra_mapping, allowed_matches)


    x = df_temp[x_temp_ref]

    if len(x.shape)==2:
        x = np.mean(x,0)
    x_spacy = df_ref['_x_raw']
   
    inputs0 = tr.tensor(df_ref['encoded']).unsqueeze(0).to(device)
    inputs0 = inputs0 if 'bert' in ref_model_name else inputs0.T
    y0 = model(inputs0).detach().numpy() 

    # sort and flip in zuco space
    # small to large
    if ignore_filt == [1,-1]:
        x[0] = np.inf 
        x[-1]  = np.inf 
        
    if random_order==False:
        if  flip_case=='generate':
            inds_sorted = np.argsort(x)[::-1] 
        elif flip_case=='destroy':
            inds_sorted =  np.argsort(np.abs(x))
        else:
            print('Select either "generate" or "destroc" reduction cases')
            raise
    else:
        inds_sorted = np.argsort(x)[::-1] 
        first_ = inds_sorted[:2]
        remain_inds = inds_sorted[2:]
        np.random.shuffle(remain_inds)
        if  flip_case=='generate':
            inds_sorted = np.array(first_.tolist() + remain_inds.tolist() )
        elif flip_case=='destroy':
            inds_sorted =  np.array(remain_inds.tolist() + first_.tolist()[::-1])
        else:
            print('Select either "generate" or "destroc" reduction cases')
            raise
            
        assert len(inds_sorted) == len(x)
            
    vals = x[inds_sorted]

    y_true = df_ref['labels']

    mse = []
    evidence = []
    model_outs = {'sentence': (tokens_raw, tokens_zuco), 'y_true':y_true, 'y0':y0}

    N=len(x)
    
    evolution = {}
    for frac in fracs:
        inds_generator = iter(inds_sorted)
        n_flip=int(np.ceil(frac*N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]
        inds_flip_spacy = [inds_map[str(i)] for i in inds_flip]
    
        if flip_case == 'destroy':
            inputs = inputs0
            for i in inds_flip_spacy:                
                if 'bert' in ref_model_name:
                    inputs[:,i] = UNK_IDX
                else:
                    inputs[i,:] = UNK_IDX
                      
        elif flip_case == 'generate':
            inputs = UNK_IDX*tr.ones_like(inputs0)
            # Set pad tokens 
            inputs[inputs0==0] = 0

            for i in inds_flip_spacy:
                if 'bert' in ref_model_name:
                    inputs[:,i] = inputs0[:,i]
                else:
                    inputs[i,:] = inputs0[i,:]

 
        y = model(inputs).detach().cpu().numpy() 
        err = np.sum((y0-y)**2)
        mse.append(err)
        evidence.append(y.squeeze()[int(y_true)])
        evolution[frac] = (inputs.detach().cpu().numpy(), inds_flip_spacy, y)
        
    if flip_case == 'generate' and frac == 1.:
        assert (inputs0 == inputs).all()
        
    model_outs['flip_evolution']  = evolution
    return mse, evidence, model_outs



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def collect_softmax_outs(x, label_map):
    words = np.array(x['sentence'][0])
    e = x['flip_evolution']
    y_true = x['y_true']
    ns = sorted(e.keys())

    E = []
    for i, frac in enumerate(ns):
        e_ = e[frac]
        out = e_[2]
        out_soft = softmax(out.squeeze())
        E.append(out_soft[label_map[int(y_true)]])
        
    return E


