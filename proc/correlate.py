import numpy as np
import itertools
from eval_utils import filter_sent, check_if_valid, get_all_keys, get_last_keys, collect_all_for_x_last, compute_corr_score, proc_attribution
import pickle
import os
import scipy.stats
chisqprob = lambda chisq, df: scipy.stats.chi2.sf(chisq, df)
from general_utils import get_analysis_config
import pandas as pd

# Standard cased to include in ranking analysis
include_rows = [
     'bnc_freq',
     'cnn0.50',
     'sattn_rels_0.25',
     'base_bert_flow_0',
     'base_bert_flow_5',
     'base_bert_flow_11',

     'fine_bert_flow_11',
     'large_bert_flow_23',
     'fine_large_bert_flow_23',

     'base_roberta_flow_0',
     'base_roberta_flow_5',
     'base_roberta_flow_11',

     'base_t5_flow_0',
     'base_t5_flow_5',
     'base_t5_flow_11',

     'bert_last',
     'fine_bert_last',
     'large_bert_last',
     'fine_large_bert_last',
     'roberta_last',
     't5_last',

     'ez_nr']


def get_attribution(df_in, k, cfg):
    x = df_in[k][cfg[k][1]]
    x = [x_ for x_ in x]
    return x


def ranking_data(df_in, task, ignore_filt=[1, -1]):
    cases_toks = []
    cases_sent = []

    cfg = get_analysis_config(task)
    X_s1 = get_attribution(df_in, 'tsr', cfg)  
    X_s1 = proc_attribution(X_s1, ignore_filt, 'tsr')
    X_c1 = np.concatenate(X_s1, axis=1).squeeze()

    keys_ = df_in.keys()

    for k in keys_:
        print(k)

        if '_last' in k:
            df_in_ = collect_all_for_x_last(df_in[k], np.mean) 
            df_in[k] = df_in_

        X_s2 = get_attribution(df_in, k, cfg )  
        X_s2 = proc_attribution(X_s2, ignore_filt, k)

        hilf_all =  [compute_corr_score(x1.squeeze(),x2.squeeze(), func=scipy.stats.pearsonr) for x1,x2 in zip(X_s1,X_s2)]
        hilf = [h[:2] for h in hilf_all]

        X_s2 = [h[3] for h in hilf_all]
        X_c2= np.concatenate(X_s2, axis=1).squeeze()

        assert len(X_c2) == len(X_c1)

        # Pearson
        hilf =  [compute_corr_score(x1.squeeze(),x2.squeeze(), func=scipy.stats.pearsonr)[:2] for x1,x2 in zip(X_s1,X_s2)]
        r_pearson_sent, p_pearson_sent = np.nanmean([x_[0] for x_ in hilf]), np.array([x_[1] for x_ in hilf])
        r_pearson_sent_std =  np.nanstd([x_[0] for x_ in hilf]),
        r_pearson_toks, p_pearson_toks = scipy.stats.pearsonr(*(X_c1, X_c2))

        #Spearman
        hilf =  [compute_corr_score(x1.squeeze(),x2.squeeze(), func=scipy.stats.spearmanr)[:2] for x1,x2 in zip(X_s1,X_s2)]
        r_spearman_sent, p_spearman_sent = np.nanmean([x_[0] for x_ in hilf]), np.array([x_[1] for x_ in hilf])
        r_spearman_sent_std =  np.nanstd([x_[0] for x_ in hilf]),
        r_spearman_toks, p_spearman_toks = scipy.stats.spearmanr(*(X_c1, X_c2))

        # Fisher test statistic for sentences
        nan_mask = ~np.isnan(p_pearson_sent)
        dgf = 2*int(nan_mask.sum())
        chi_2 = -2*np.sum(np.log(p_pearson_sent[nan_mask]))
        p_pearson_sent = chisqprob(chi_2, dgf)

        nan_mask = ~np.isnan(p_spearman_sent)
        dgf = 2*int(nan_mask.sum())

        chi_2 = -2*np.sum(np.log(p_spearman_sent[nan_mask]))
        p_spearman_sent = chisqprob(chi_2, dgf)

        k_clean = k.replace('_tsr_x', '').replace('tsr_x_', '')
        cases_toks.append([k_clean, r_pearson_toks, p_pearson_toks,  r_spearman_toks, p_spearman_toks])
        cases_sent.append([k_clean,  r_pearson_sent, p_pearson_sent, r_pearson_sent_std ,
                           r_spearman_sent, p_spearman_sent, r_spearman_sent_std])

    dfs = {'token':pd.DataFrame(cases_toks, columns=['names', 'pearson', 'p_pearson', 'spearman',  'p_spearman']),
           'sentence': pd.DataFrame(cases_sent, columns=['names', 
                                                         'pearson', 'p_pearson', 'std_pearson',
                                                         'spearman',  'p_spearman', 'std_spearman',
                                                         ])}

    return dfs
    
    
def collect_ranking_over_seeds(df_template_dir, task, seeds=[1]):
    
    dfs_seeds = {i: [] for i in seeds}
    for seed in seeds:
        df_in = pickle.load(open(os.path.join(df_template_dir.format(seed), 'dfs_all_{}.p'.format(task)), 'rb'))
        dfs = ranking_data(df_in, task)
        dfs_seeds[seed] = dfs
        
    return dfs_seeds


def average_dfs(dfs, cols=['pearson', 'p_pearson', 'spearman', 'p_spearman']):
    
    for i, df in enumerate(dfs):
        names = list(df.names)
        df = df.set_index('names')
        if i == 0:
            data = {c: {n: [] for n in names} for c in cols}
            df_ = pd.DataFrame([[n, [], [], [], []] for n in names],  columns=['names']+cols)
            df_ = df_.set_index('names')
        for c in cols:
            for name in names:
                val = df[c][name].item()
                data[c][name].append(val)
                df_.loc[name, c].append(val)
                
    df_avg = pd.DataFrame([[n]+[None]*6 for n in names], 
                          columns = ['names']+['pearson', 'p_pearson','std_pearson',
                                               'spearman', 'p_spearman', 'std_spearman'])
   
    df_avg['names_'] = list(df_avg['names'])
    df_avg = df_avg.set_index('names_')

    for name in names:
        for c in cols:
            vals = df_.loc[name, c]
            if c.startswith('p_'):
                chi_2 = -2*np.sum(np.log(vals))
                val_ = chisqprob(chi_2, 2*len(vals))
                
                df_avg.loc[name, c] = val_ #np.max(vals)
            else:
                df_avg.loc[name, c] = np.mean(vals)
                df_avg.loc[name, 'std_'+c] = np.std(vals)    
                
    return data, df_, df_avg

