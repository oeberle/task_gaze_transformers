name_map = {'sattn': 'self-attention',
            'sattn_rels_0.25': 'self-attention (LRP)',
            'cnn0.50': 'CNN (LRP)',
            'tsr': 'TSR (ZuCo)',
            'ez_nr': 'E-Z Reader',

            'bnc_freq': 'BNC inv prob',
            'bnc_freq_prob': 'BNC prob',
            'random': 'Random',
            # 'bert': 'bert_mean',

            'bert': 'BERT mean',
            'fine_bert': 'fine-BERT mean',
            'large_bert': 'large-BERT mean',
            'fine_large_bert': 'fine-large-BERT mean',

            'bert_max': 'BERT max',
            'fine_bert_max': 'fine-BERT max ',

            "fine_bert_flow_0": "fine-BERT flow 0",
            "fine_bert_flow_5": "fine-BERT flow 5",
            "fine_bert_flow_11": "fine-BERT flow 11",

            "base_bert_flow_0": "BERT flow 0",
            "base_bert_flow_5": "BERT flow 5",
            "base_bert_flow_11": "BERT flow 11",

            'bert_large': 'BERT-large mean',
            'fine_large_bert': 'fine-BERT-large mean',

            'large_bert_max': 'BERT-large  max',
            'fine_large_bert_max': 'fine-BERT-large max ',

            "fine_large_bert_flow_0": "fine-BERT-large flow 0",
            "fine_large_bert_flow_11": "fine-BERT-large flow 11",
            "fine_large_bert_flow_23": "fine-BERT-large flow 23",

            "large_bert_flow_0": "BERT-large flow 0",
            "large_bert_flow_11": "BERT-large flow 11",
            "large_bert_flow_23": "BERT-large flow 23",
            
            
            "t5": "T5 mean",
            "t5_max": "T5 max", 
            "base_t5_flow_0": "T5 flow 0",
            "base_t5_flow_5": "T5 flow 5",
            "base_t5_flow_11": "T5 flow 11",
            
                        
            'roberta': 'RoBERTa mean',
            "roberta_max": "RoBERTa max", 
            "base_roberta_flow_0": "RoBERTa flow 0",
            "base_roberta_flow_5": "RoBERTa flow 5",
            "base_roberta_flow_11": "RoBERTa flow 11",
            

            "t5_last": "T5 mean", 
            "roberta_last": "RoBERTa mean", 
            'large_bert_last': 'BERT-large mean',
            'fine_large_bert_last': 'fine-BERT-large mean ',
            'bert_last': 'BERT mean',
            'fine_bert_last': 'fine-BERT mean ',

            
            }

replace_rules = [('flow_0_x_flow_0', 'flow_0'), 
                ('flow_5_x_flow_5', 'flow_5'),
                ('flow_11_x_flow_11', 'flow_11'),
                 
                ('flow_23_x_flow_23', 'flow_23'),
                ( 'max_x_max', 'max'),
                ( 'last_x_last', 'last'),

                 ('_x','')]



def proc_str(x, replace_rules):
    for t0, t1 in replace_rules:
        x = x.replace(t0,t1)
    return x


def proc_name(x,replace_rules, name_map):
    x_ = proc_str(x, replace_rules)
    x_ = x_.replace('_abs','').replace('_TT','')
    return name_map[x_] if x_ in name_map else x_


def get_color(key):
    '''https://www.colorhexa.com/'''
    lw = 2
    if key == 'random':
        c = 'black'
        ls = '-'
    elif key == 'tsr':
        c = '#ac37ff'  # plum
        ls = '-'
        lw = 2
    elif 'ez_' in key:
        c = '#cc0000'  # red
        ls = '-'
        lw = 2
    elif 'fine_bert' in key or 'fine_large_bert' in key:
        c = '#1755c4'  # dark blue
        ls = '-'
    elif 'roberta' in key:
        c = '#5adada'#812e81'  # lime
        ls = '-'
    
    elif 'bert' in key:
        c = '#87adf1'  # cornflower
        ls = '-'
    elif 'bnc_freq_prob' == key:
        c = '#828282'  # gray
        ls = '--'
    elif 'bnc_freq' == key:
        c = '#828282'  # gray
        ls = '-'
    elif 'cnn' in key:
        c = '#f5d033'#ff9000'  # hot pink
        ls = ':'
    elif 'sattn_rels_' in key:
        c = '#f2835f'
        ls = ':'
    elif 'sattn' == key:
        c = '#5e8746'#5bc138'  # lime
        ls = '-'
    elif 't5' in key:
        c = '#0a7b8a' #c199c1'
        ls = '-'
    else:
        print(key)
        c = 'black'  # lime
        ls = '-'
      #  raise KeyError(f"no color defined for key {key}")
    return c, ls, lw


label_dict_wikirel = {'award': 0, 'education': 1, 'employer': 2, 'founder': 3,
                      'job_title': 4, 'nationality': 5,
                      'political_affiliation': 6, 'visited': 7, 'wife': 8}
class_dict_wikirel = {v: k for k, v in label_dict_wikirel.items()}

class_dict_sst = {0: 'negative', 1: 'neutral', 2: 'positive'}
label_dict_sst = {v: k for k, v in class_dict_sst.items()}

figsize_width_2col = 8
figsize_width_1col = figsize_width_2col / 2
