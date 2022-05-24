import numpy as np
import pandas as pd
from Levenshtein import distance
import warnings
# warnings.filterwarnings("ignore")

def remove_include_toks(t_list, include_list):
    
    if len(include_list)>0:
        t = []
        inds = []
        for i, t_ in enumerate(t_list):
            for item in include_list:
                if item in t_:
                    t_proc =  t_.replace(item, '')
                    inds.append(i)
                    break
                else:
                    t_proc = t_

            t.append(t_proc)
    else:
        t = t_list
        inds = []

    return t, inds
                

def replace_special_toks(t_list, extra_mapping):
    t = []
    for t_ in t_list:
        for item in sorted(extra_mapping.keys(), key = lambda x: len(x))[::-1]:
            if item in t_:
                t_proc =  t_.replace(item, extra_mapping[item])
                break
            else:
                t_proc = t_
        t.append(t_proc)
    return t     
    
    
def get_join_inds(tokens_raw, tokens_zuco, extra_mapping={}, allowed_matches={}):
    
    
    tokens_raw = [t.lower() for t in tokens_raw]  
    tokens_raw = replace_special_toks(tokens_raw, extra_mapping)
    tokens_zuco = replace_special_toks(tokens_zuco, extra_mapping)   
    tokens_zuco = [t.lower() for t in tokens_zuco]
    
    
    tokens_raw_gen = iter(enumerate(tokens_raw))
    tokens_raw_list = []
    inds_map = {}
    
    for i, w in enumerate(tokens_zuco):
        w_rec = ''
        sublist = []
        sublist_inds = []
        
        while w != w_rec:
            ind, tok = next(tokens_raw_gen)

            sublist.append(tok)
            sublist_inds.append(ind)
            w_rec = ''.join(sublist)

            # Check if extra_mapping is fulfilled
            if w_rec in allowed_matches:
                if allowed_matches[w_rec] == w:
                    w_rec=w     
                elif distance(w, w_rec) <= 3:
                    print('Distance matched', w, w_rec)
                    w_rec=w  
                else:
                    print('Alignment of sentences not successful:\n{}\n{}'.format(tokens_zuco, tokens_raw))
                    return None, None, None
            
        tokens_raw_list.append(sublist)
        inds_map[str(i)] = sublist_inds
        
    for j, remaining_words in tokens_raw_gen:
        i+=1
        tokens_raw_list.append([remaining_words])
        inds_map[str(i)+'*'] = [j]
  
    
    return tokens_zuco, tokens_raw_list, inds_map

class SentenceAlignment(object):
    """
    Alginment of sentences between experimental and (tokenizer)-processed data, i.e.

   ["It's", 'everything', 'you', "don't", 'go', 'to', 'the', 'movies', 'for.']  
   ['it', "'s", 'everything', 'you', 'do', "n't", 'go', 'to', 'the', 'movies', 'for', '.', '</s>'].

    """    
   
    def __init__(self, human_df, x='x', match_threshold=12, remove_list=['<pad>'], 
                 include_list=['</s>'], extra_mapping={}, allowed_matches={}):
        
        """
        Initialize and prepare alginment of sentences.

        Args:
            human_df: Used as the reference dataframe to align sentenes to.
            x: Determines the reference attribution column to be used, i.e. 'x' or 'x_flow_11'
            match_threshold: Allowed difference in characters to consider alignment.
            remove_list: Tokens that are ignored for the aligment, i.e. <pad>.
            include_list: Tokens that are included for the aligment, i.e. </s>.
            extra_mapping: Dictionary that contains a token-to-token mapping between tokens in sentence1 
            and sentence2  which is useful to allow, i.e. aligment of <unk> tokens added by the tokenizer.
            allowed_matches: Dictionary that contains a token-to-token mapping between tokens sentence1 
            and sentence2 which is useful to allow non-exact alignment which can be caused by tokenizer rules.
        """
        self.human_df = human_df
        self.human_texts, self.human_attns = self.proc_human_df()
        self.ids2 = list(self.human_texts.keys())
        self.match_threshold = match_threshold
        self.remove_list = remove_list
        self.include_list = include_list
        self.extra_mapping = extra_mapping
        self.allowed_matches = allowed_matches
        self.x = x
        self.model_matching = True
        
        
    def proc_human_df(self):
        human_texts = {id_:''.join([x_.lower() for x_ in row.words]) for id_,row in self.human_df.iterrows()}
        human_attns = {id_: np.nan_to_num(row.x) for id_,row in  self.human_df.iterrows()}      
        return human_texts, human_attns
    
    def remove_toks(self, words, attn, remove_list):
        words_attn = [(w,a) for w,a in zip(words, attn) if w not in remove_list]
        words, attn = [w[0] for w in words_attn], [w[1] for w in words_attn]

        control = [a for w,a in zip(words, attn) if w in remove_list]
        assert np.sum(control) == 0.
        
        return words, attn

    def match_single_sentence(self, textstr):
        edit_dist = [distance(textstr,self.human_texts[tz]) for tz in self.ids2]
        min_idx = np.argmin(edit_dist)
        min_val = edit_dist[min_idx]
        id2 = self.ids2[min_idx]  
        return min_idx, min_val, id2


    def proc_ez_matching(self, df, df_ref):
        # sort df like df_ref
        if len(df) != len(df_ref):
            warnings.warn("Data frames have different lengths")
        ez_texts = {id_: ''.join([x_.lower() for x_ in row.words]) for id_, row in df.iterrows()}
        ids1 = list(ez_texts.keys())
        df_all_keys = list(df.index)
        idx = []
        remove_ids_df_ref = []
        
        for id2, v in df_ref.iterrows():
            textstr = ''.join(list(v.words)).replace('</s>', '').replace('<unk>', '').replace('<pad>', '').\
                replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '')
            
            edit_dist = [distance(textstr, ez_texts[tz]) for tz in ids1]
            min_idx = np.argmin(edit_dist)
            min_val = edit_dist[min_idx]
            id1 = ids1[min_idx]  
            
            if min_val <= 12:
                idx.append([id1, id2])
                if id1 in df_all_keys:
                    df_all_keys.remove(id1)
                else:
                    remove_ids_df_ref.append([id1, id2])
            else:
                remove_ids_df_ref.append([id1, id2])
             
        remove_ids_df = df_all_keys
        return idx, remove_ids_df_ref, remove_ids_df

    def map_dfs(self, df_model):
        merge_func = np.max  
        min_dists, min_idxs = [], []

        # mapping idx_df_human:idx_df_model
        idx_map = {}

        if self.x == 'x_last':
            num_layers = np.max([int(k.split('_')[1]) for k in df_model.keys()
                                 if len(k.split('_')) == 3 and 'flow' not in k and 'rollout' not in k])
            all_x_max = [k for k in df_model.keys()
                         if len(k.split('_')) == 3
                         and 'flow' not in k
                         and 'rollout' not in k
                         and int(k.split('_')[1]) == num_layers]
            all_x_max_raw = [key + '_raw' for key in all_x_max]
            df_model_proc = pd.DataFrame(columns=['_tokens_raw', '_x_raw', 'words', self.x, 'x',
                                                  self.x + '_raw', 'model_output', 'ypred', 'tokens_raw_list'] +
                                         all_x_max + all_x_max_raw)
        elif self.x != 'x':
            df_model_proc = pd.DataFrame(columns=['_tokens_raw', '_x_raw', 'words', self.x, 'x',
                                                  self.x + '_raw', 'model_output', 'ypred', 'tokens_raw_list'])

        else:

            df_model_proc = pd.DataFrame(columns=['_tokens_raw', '_x_raw', 'words', self.x,
                                                  self.x + '_raw', 'model_output', 'ypred', 'tokens_raw_list'])

        self.x_init = self.x

        if self.x == 'x_last':
            # set keys for all layers/heads
            self.x = all_x_max[0]  # set an example key

        for id1, row in df_model.iterrows():

            toks = row.tokens
            attns = np.array(row[self.x]).squeeze()
            if 'output' in df_model.keys():
                model_output = row.output
            else:
                model_output = [None]

            ypred = row.ypred if 'ypred' in df_model.keys() else [None]

            t = [w for w in toks if w not in self.remove_list]

            textstr = ''.join(list(t)).replace('</s>', '').replace('<unk>', '').replace('<pad>', ''). \
                replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '')

            min_idx, min_val, id2 = self.match_single_sentence(textstr)

            # First align then join
            tokens_zuco = [w.lower() for w in list(self.human_df.loc[id2].words)]

            # Remove tokens that are added by tokenizer to align to zuco sentences
            tokens_raw, _ = remove_include_toks(t, self.include_list)  # e.g. "</s>"

            # Adding end of sentence token to zuco
            if self.model_matching:
                tokens_zuco[-1] = tokens_zuco[-1] + tokens_raw[-1]

            _, tokens_raw_list, inds_map = get_join_inds(tokens_raw, tokens_zuco,
                                                         extra_mapping=self.extra_mapping,
                                                         allowed_matches=self.allowed_matches)

            t = [''.join(t_) for t_ in tokens_raw_list]

            if len(attns.shape) == 1:  # Single-attention vector (e.g. BNC)
                attns_raw = attns
                a = np.array([merge_func([attns_raw[ix] for ix in inds_map[str(ind)]])
                              for ind in sorted(map(int, inds_map.keys()))])
            elif len(attns.shape) == 2:  # Multi-attention matrix (e.g. shallow self-attention)
                attns_raw = attns
                a = np.array([[merge_func([a_[ix] for ix in inds_map[str(ind)]]) for a_ in attns_raw]
                              for ind in sorted(map(int, inds_map.keys()))]).T

            df_model_proc.loc[id1, 'tokens_raw_list'] = np.array(tokens_raw_list, dtype=object)

            if self.x_init != 'x_last':
                df_model_proc.loc[id1, self.x] = a
                df_model_proc.loc[id1, self.x + '_raw'] = attns_raw
            df_model_proc.loc[id1, 'words'] = t
            df_model_proc.loc[id1, 'model_output'] = model_output
            df_model_proc.loc[id1, 'ypred'] = ypred
            df_model_proc.loc[id1, '_tokens_raw'] = df_model.loc[id1, 'tokens']
            df_model_proc.loc[id1, '_x_raw'] = df_model.loc[id1, 'x']

            # Do it for all other keys
            # all layers/heads are stored in df
            if self.x_init == 'x_last':
                for key in all_x_max:
                    attns = np.array(row[key]).squeeze()
                    a = np.array([merge_func([attns[ix] for ix in inds_map[str(ind)]])
                                  for ind in sorted(map(int, inds_map.keys()))])
                    df_model_proc.loc[id1, key] = a
                    df_model_proc.loc[id1, key + '_raw'] = attns

            if self.x != 'x':
                df_model_proc.loc[id1, 'x'] = a  # Copy for flipping

            min_dists.append(min_val)
            idx_map[id1] = id2
            min_idxs.append(id2)

        df_model_proc['labels'] = df_model['labels']
        df_model_proc['encoded'] = df_model['encoded']
        if 'output' in df_model:
            df_model_proc['output'] = df_model['output']

        return idx_map, df_model_proc

    def compare_df_to_human(self, df):
        # this basically just checks all is matched
        idx_map, df_proc = self.map_dfs(df)
        self.df_proc = df_proc
        idx = []
        data = []
        for id1, id2 in idx_map.items():
            
            attn1, words1 = df_proc.loc[id1, self.x], df_proc.loc[id1, 'words']
            attn2, words2 = self.human_attns[id2], self.human_df.words.loc[id2]
                
            # Check valid aligment
            lattn1 = attn1.shape[-1]
            lattn2 = attn2.shape[-1]
            assert lattn1 == lattn2

            dat = [(attn1, words1), (attn2, words2)]
            data.append(dat)
            idx.append([id1, id2])

        print('Matched {}/{}'.format(len(data), len(idx_map)))
        
        # Sort by sorted reference index
        ids2 = list(map(list, zip(*idx)))[1]
        ixs = np.argsort(ids2)
        data = [data[i] for i in ixs]
        idx = [idx[i] for i in ixs]
        return data, idx

    def compare_two_model_dfs(self, df1, df2):
        idx_map = {k: v for k, v in zip(df1.index, df2.index)}
        data = []
        idx = []
        for id1, id2 in idx_map.items():
            
            dat1, dat2 = df1.loc[id1], df2.loc[id2]
            attn1, words1 = dat1.x.squeeze(), dat1.tokens
            
            attn2, words2 = dat2.x.squeeze(), dat2.tokens
            
            #Remove padding 
            words1, attn1 = self.proc_join(words1, attn1)
            words2, attn2 = self.proc_join(words2, attn2)
   
            assert (np.array(words1) == np.array(words2)).all()

            dat = [(attn1, words1), (attn2, words2)]
            data.append(dat)
            idx.append([id1, id2])

        return data, idx