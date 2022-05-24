import pandas as pd
import numpy as np
import scipy.stats

def proc_rel_abs(x):
    x = np.array(x)
    if np.min(x)<0.:
        return np.abs(x)
    else:
        return x
    
    
def sum_normalize(x):
    return x/np.sum(abs(x))


def min_max_normalize(x):
    assert(np.min(x) >= 0)
    return (x-np.min(x))/(np.max(x)-np.min(x))


def get_human_avg_df(df_human, ignore_zeros=False):
    def vector_average(group, ignore_zeros):
        assert (np.nan not in np.array(group.duration.tolist()))
        series_to_array = np.array(group.duration.tolist())
        if ignore_zeros:
            print('Ignore zeros in avg')
            series_to_array[series_to_array == 0] = np.nan
        words = group.words.tolist()
        assert (np.array(words)[0] == np.array(words)).all()
        text_id = group.text_id.tolist()[0]
        labels = group.labels.tolist()[0]
        ypred = 'N/A'
        #nan_to_num if all attns at one idx for all subjects: nan -> 0.
        attns = np.nan_to_num(np.nanmean(series_to_array, axis = 0))
        return text_id, words[0], attns, labels, ypred

    attn_subj_avg = df_human.groupby(['text_id','labels']).apply(vector_average, ignore_zeros=ignore_zeros)
    df_human_avg = pd.DataFrame(attn_subj_avg.tolist(), 
                   columns =['text_id', 'words', 'x', 'labels', 'ypred']) 
    return df_human_avg


def create_df(words, weights):
    area_width = 10
    area_height = 8
    D = []
    i = 0
    area_left_x = 0.
    area_right_x = 10.

    area_top_y = 0.
    area_bottom_y = 5.

    for w, heat in zip(words, weights):
        area_left_x += 10  # *len(w)
        area_right_x += 0  # *len(w)

        D.append([i, w, area_width, area_height, area_left_x, area_bottom_y, area_right_x, area_top_y, heat])
    df = pd.DataFrame(D, columns=['text_id', 'word', 'area_width', 'area_height', 'area_left_x', 'area_bottom_y',
                                  'area_right_x', 'area_top_y', 'heat'])
    return df


    

def get_all_keys(df):
    keys = [k for k in df.keys() if len(k.split('_'))==3 and 'flow' not in k and 'raw' not in k and 'rollout' not in k]
    assert len(keys) in [12*12,24*16, 12, 16]
    return keys


def get_last_keys(df):
    keys = get_all_keys(df)
    last_layer = np.max([int(k.split('_')[-2]) for k in keys])
    last_keys = [k for k in keys if int(k.split('_')[-2])==last_layer]
    return last_keys
    

def collect_all_for_x_last(df, select_func=np.mean):
    # df: contains all relative attentions from each layer and head
    last_keys = get_last_keys(df)
    df['x_last'] = np.array([np.nan]*len(df)).astype(object)

    for id_, row  in df.iterrows():
        # collect attns for each head
        attention_mat = np.array([df.loc[id_,k] for k in last_keys])

        # Averaging over attention heads (Abnar & Zuidema 2020)
        attention_ = select_func(attention_mat, axis=0)
        df.loc[id_,'x_last'] = attention_
    return df

    
def compute_corr_score(x1, x2, case='max', func=scipy.stats.pearsonr, debug=False):

    if debug:
         import pdb;pdb.set_trace()

    # Make sure that x1 is a 1d-vector
    if len(x1.shape) ==2:
        raise
    elif len(x2.shape) ==2:
        cs = [func(x2_,x1)[0] for x2_ in x2]
        imax = np.argmax(cs)
        c = cs[imax]
        if case == 'max':
            imax = np.argmax(cs)
            c = cs[imax]
            x2_ = x2[imax]
        elif case == 'min':
            imax = np.argmin(cs)
            c = cs[imax]
            x2_ = x2[imax]
        elif case == 'mean':
            c = np.mean(cs)
            x2_ = np.mean(x2, axis=0)
        pval = func(x2_,x1)[1]
        return c,pval, x1, x2_[np.newaxis,:]

    else:
        c, pval = func(x1,x2)
        return c,pval, x1, x2[np.newaxis,:]

    
def proc_attribution(X, ignore_filt, case):
    X_ = []
    for x in X:
        assert np.nanmin(x)>=0.

        # Filter start/end of sentence
        try:
            x_ = filter_sent(x, ignore_filt)
        except AttributeError:
            import ipdb;ipdb.set_trace()
        
        # Filter attributions that only contain one word
        if not check_if_valid(x_):
            continue

        if len(np.shape(x_)) == 1:
            x_ = x_[np.newaxis, :]

        X_.append(x_)
        
    return X_

 
def add_proc_funcs(df, proc_func_dict, column='x'):
    new = []
    for k, func in proc_func_dict.items():
        df[column + '_' + k] = df[column].apply(func)
        new.append(column + '_' + k)
    return df, new


def filter_sent(a, filt=[0, None]):
    if len(a.shape) == 1:
        return a[filt[0]:filt[1]]
    elif len(a.shape) == 2:
        return a[:, filt[0]:filt[1]]

    
def check_if_valid(a):
    if len(a.shape) == 1:
        return True if len(a)>1 else False
    elif len(a.shape) == 2:
        return True if a.shape[1]>1 else False


def filter_duplicates(df1):
    if len(list(df1.index)) != len(list(set(df1.index))):
        # Skipping duplicate sentences (same from the model perspective)
        df1 = df1[~df1.index.duplicated(keep='first')]
    return df1


def get_matching_vars(case, task):

    if 'roberta' in case:
        include_list = ['<s>','Ġ']
        remove_list = ['<pad>']
    elif 'bert' in case:
        include_list = ['[CLS]','##']
        remove_list = ['[PAD]']
    elif 't5' in case:
        include_list = ['<s>','▁']
        remove_list = ['<pad>']
    else:
        include_list = []
        remove_list = ['<pad>']
    
    allowed_matches = {'---' : '--',
                       '``...' : '...'}
    
    if task=='SR':
        if 'roberta' in case:
            extra_mapping = {'Ġ':''}
        elif 'bert' in case:
            extra_mapping = {'##':'', '\\':''}
        elif 't5' in case:
            extra_mapping = {'▁':'',
                             "``" : "<unk>",
                             "`" : "<unk>"}
        else:
            extra_mapping = {'\\':'',
                             'emp11111ty':'<unk>',
                             'film.1': '<unk>',
                             "i'm": 'i<unk>',
                             "i've": 'i<unk>',
                             "you've": 'you<unk>',
                             'a**holes':'<unk>',
                             "hardy'n":'<unk>',
                             'co-writer/director': 'co-writer<unk>director',
                             'action/thriller': 'action<unk>thriller',
                             'mother/daughter': 'mother<unk>daughter',
                             '(unfortunately': '<unk>unfortunately',
                             'r-rated)': 'r-rated<unk>'}
    elif task in ['TSR']:
        if 'roberta' in case:
            extra_mapping = {'Ġ':'',
                             'km�':'kmï¿½'}
        elif 'bert' in case:
            extra_mapping = {'##':'',
                            '\\':'',
                            'km�':'km'}   
        elif 't5' in case:
            extra_mapping = {'▁':'',
                             "``" : "<unk>",
                             "`" : "<unk>",
                             'km�':'km'}
        else:
            extra_mapping = {'\\':'',
                             'wuerttemberg': '<unk>',
                             'km�':'<unk><unk>',
                             'bucher': '<unk>',
                             'duarte(october':'<unk>',
                             'erdogan':'<unk>',
                             'tuebingen':'<unk>',
                             'do': '<unk>',
                             '1606-april': '<unk>',
                             '/?g?nz?b?g/': '<unk>',
                             'brunnhilde': '<unk>',
                             'jose': '<unk>',
                             'cintron':'<unk>',
                             '1890-december':'<unk>',
                             '1universidad': '<unk>',
                             'sequel': '<unk>',
                             'rene' : '<unk>'}

            
    return remove_list, include_list, extra_mapping, allowed_matches

