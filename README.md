# Do Transformer Models Show Similar Attention Patterns to Task-Specific Human Gaze?
Code for "Do Transformer Models Show Similar Attention Patterns to Task-Specific Human Gaze?" paper (ACL 2022)
https://aclanthology.org/2022.acl-long.296.pdf



## First Steps
### `_local_options.py`
Create a file `_local_options.py` with one line `data_root = ...` with the 
path to the directory containing the textsaliency data, e.g. 
`"/home/space/textsaliency"`. Depending on which scripts you may run, this 
directory has to contain up to three subdirectories: `datasets`, `experiments`, 
and `models`. 

## Data
Download the data for our analysis using `get_osf_data.sh'. The resulting folders contain all data necessary to reproduce our results.


Added   
`model_output`, `eval_folder`, `zuco_files` where paths are added for dfs going 
into run_correlation, run_analysis and where the zuco files are stored

## Scripts
* `run_analysis.py`: runs correlation analyses, most important functions are in 
the analysis folder, necessary to set flags (pos, labels,sen_len, word_len, 
word_prob)
* `run_alignment.py` can be called with `--task {SR, TSR}` alone, or add explicit yaml file

* `run_flipping.py`: compute the input reduction analysis using the output dataframe (df_all_file) from `run_correlation.py`. Specify `--df_all_file ../dfs_all_{SR,TSR}.p `, `--analysis_file configs/analysis/{sst, wikirel}_base_pub.yaml`, `--config_file MODEL_CONFIG` and `--task {SR, TSR}`. Output files are used in `run_analysis.py` for the input reduction (Alternatively you can use the default files in `data/all_flip_cases_{SR, TSR}.p`).

__Please note:__ Results can deviate from the plots in the paper based on respective package versions, in particular when using spacy 3 (the paper shows results for spacy 2.3.2)


## Cite

    @inproceedings{eberle-etal-2022-transformer,
        title = "Do Transformer Models Show Similar Attention Patterns to Task-Specific Human Gaze?",
        author = "Eberle, Oliver  and
          Brandl, Stephanie  and
          Pilot, Jonas  and
          S{\o}gaard, Anders",
        booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        month = may,
        year = "2022",
        address = "Dublin, Ireland",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2022.acl-long.296",
        pages = "4295--4309",
    }
