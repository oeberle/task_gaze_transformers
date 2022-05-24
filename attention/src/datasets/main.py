from .sst import SST_Dataset
from .zuco_nlp import Zuco_NLP_Dataset
from .wiki_relations import WikiRelations

def load_dataset(dataset_name, data_path, tokenizer='spacy', use_tfidf_weights=False,
                 append_sos=False, append_eos=False, clean_txt=False, encoder=None, task='TSR'):
    """Loads the dataset."""


    implemented_datasets = ('sst', 
                            'sst_without_zuco',
                            'wiki_rel',
                            'wiki_rel_without_zuco',
                            'zuco_sst',
                            'zuco_wiki_rel',
                            'zuco_nr')

    assert dataset_name in implemented_datasets

    dataset = None
    
    if dataset_name == 'sst':
        dataset = SST_Dataset(root=data_path, tokenizer=tokenizer,
                              use_tfidf_weights=use_tfidf_weights, append_sos=append_sos, append_eos=append_eos,
                              clean_txt=clean_txt, filter_out_zuco=False, keep_only_zuco=False, encoder=encoder)
        
    elif dataset_name == 'sst_without_zuco':
        dataset = SST_Dataset(root=data_path, tokenizer=tokenizer,
                              use_tfidf_weights=use_tfidf_weights, append_sos=append_sos, append_eos=append_eos,
                              clean_txt=clean_txt, filter_out_zuco=True, keep_only_zuco=False,encoder=encoder)    
        

    elif dataset_name == 'wiki_rel':
        dataset = WikiRelations(tokenizer=tokenizer, encoder=encoder, zuco_subset=True, filter_out_zuco=False, clean_txt=clean_txt)
       
    elif dataset_name == 'wiki_rel_without_zuco':
        dataset = WikiRelations(tokenizer=tokenizer, encoder=encoder, zuco_subset=True, filter_out_zuco=True, clean_txt=clean_txt)
    

    elif dataset_name == 'zuco_sst':
        # Alternativr way to load zuco_sst (gives you the removed docs from SST)
        #dataset = SST_Dataset(root=data_path, tokenizer=tokenizer,
        #          use_tfidf_weights=use_tfidf_weights, append_sos=append_sos, append_eos=append_eos,
        #          clean_txt=clean_txt, filter_zuco=True, keep_only_zuco=True,encoder=encoder)
        assert task=='SR'
        dataset = Zuco_NLP_Dataset(root='', task=task, tokenizer=tokenizer, encoder=encoder, exclude_control=True, clean_txt=clean_txt)
        
    elif dataset_name == 'zuco_wiki_rel':
        assert task=='TSR'
        dataset = Zuco_NLP_Dataset(root='', task=task, tokenizer=tokenizer, encoder=encoder, exclude_control=True, clean_txt=clean_txt)
    
    elif dataset_name == 'zuco_nr':
        assert task=='NR'
        dataset = Zuco_NLP_Dataset(root='', task=task, tokenizer=tokenizer, encoder=encoder, exclude_control=True, clean_txt=clean_txt)


    return dataset
