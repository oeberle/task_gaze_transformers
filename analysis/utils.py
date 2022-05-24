import pandas as pd
import numpy as np
import spacy
from typing import List, Tuple, Any
from scipy.stats import spearmanr

nlp = spacy.load("en_core_web_sm")

def prepare_dfs(dfs_orig: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Prepares data frames, i.e. merge bnc, fine_bert_flow_11 and ez into one df.
    Extracts second df on word-level

    Args:
        dfs_orig:
            Original pd.DataFrame as comes out of evaluation which includes
            importance attributes for all models.

    Returns:
        df (pd.DataFrame): dataframe which includes the following columns
            'text_id': unique id for each sentence
            'words': stimuli/sentences
            'labels': class labels as defined in SST and Wikipedia datasets
            'x': total fixation times averaged across all participants
            'TT': importance attributes from EZ_Reader
            'cloze': word probabilities as calculated by Kneser-Ney
            'bert_flow_11': flow values from BERT task-tuned for the last layer
            'x_bnc': frequency baseline based on BNC

        df_exploded (pd.DataFrame): ...
    """

    #iteratively merge data frames
    df_ez = pd.merge(dfs_orig['tsr'][['x', 'words']],
                     dfs_orig['ez_nr'][['TT', 'cloze']],
                     on='text_id')

    df_ez_flow = pd.merge(
        df_ez[['x', 'words', 'TT', 'cloze']].reset_index(),
        dfs_orig['base_bert_flow_11'][['x_flow_11', 'labels']].rename(
            columns={'x_flow_11': 'bert_flow_11'}).reset_index(),
        on='text_id')

    df_ez_flow_roberta = pd.merge(
        df_ez_flow.reset_index(),
        dfs_orig['base_roberta_flow_11'][['x_flow_11']].rename(columns={'x_flow_11': 'roberta_flow_11'}).reset_index(),
        on='text_id'
    )

    df_ez_flow_roberta_t5 = pd.merge(
        df_ez_flow_roberta.reset_index(),
        dfs_orig['base_t5_flow_11'][['x_flow_11']].rename(columns={'x_flow_11': 't5_flow_11'}).reset_index(),
        on='text_id'
    )

    df_bnc = dfs_orig['bnc_freq'].reset_index().rename(columns={'x': 'x_bnc'})

    df = pd.merge(
        (df_ez_flow_roberta_t5[['x','words','TT','cloze','text_id','bert_flow_11','labels','t5_flow_11','roberta_flow_11']]
         .reset_index()),
        df_bnc[['x_bnc', 'text_id']].reset_index(),
        on='text_id')

    return df, df.apply(pd.Series.explode).reset_index()

def standardize_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Normalize columns to mean=0 and std=1. New column names will contain the 
    suffix "_normalized".

    Args:
        df (pd.DataFrame): dataframe on which columns are to be normalized
        cols (list of str): names of columns to normalize

    Returns:
        pd.DataFrame: `df` with normalized columns

    """
    for col in cols:
        new_col = col + '_normalized'
        df[new_col] = (df[col] - np.mean(df[col])) / (np.std(df[col]))

    return df

def min_max_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Normalize columns to min=0 and max=1. New column names will contain the 
    suffix "_min_max_normalized".

    Args:
        df (pd.DataFrame): dataframe on which columns are to be normalized
        cols (list of str): names of columns to normalize

    Returns:
        pd.DataFrame: `df` with normalized columns

    """
    for col in cols:
        new_col = col + '_min_max_normalized'
        assert df[col].min() >= 0
        df[new_col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df

def extract_pos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Part-of-Speech Tag and add column to input dataframe.

    Args:
        df: pd.DataFrame on word-level from which POS is to be extracted

    Returns
        pd.DataFrame: `df` with additional column `pos`
    """
    for index, row in df.iterrows():
        if np.isnan(df.loc[index, 'TT']):
            continue
        doc = nlp(row['words'])
        pos = [token.pos_ for token in doc]
        df.loc[index, 'pos'] = pos[0]

    return df

def extract_sen_len(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract sentence length and add column to input dataframe

    Args:
        df (pd.DataFrame): 
            dataframe on sentence-level on which sentence length is to be 
            extracted

    Returns
        pd.DataFrame: `df` with additional column `sen_len`
    """
    for index, row in df.iterrows():
        df.loc[index, 'sen_len'] = len(row['x'])
    return df

def extract_word_len(df: pd.DataFrame):
    """
    Extract word length and add column to pd.DataFrame

    Args:
        df: pd.DataFrame on word-level on which word length is extracted

    Returns
        pd.DataFrame: `df` with additional column `word_len`
    """
    for index, row in df.iterrows():
        df.loc[index, 'word_len'] = len(row['words'])
    return df

def correlate_category(df: pd.DataFrame, bin: str, bins: List[Any]=[], 
                       level: str='word', tokens: bool=True, samples: bool=True):
    """
    Correlates given category in data frame with human fixation (averaged total 
    reading time)

    Args:
        df (pd.DataFrame): 
            dataframe which includes importance scores on sentence or word-level
        bin (str): name of the column to correlate with
        bins (list): 
            pre-defined bins. ... 
        level (str, optional): 
            either word or sentence, depends on the category
        tokens (bool, optional): 
            whether attributes scores are also returned, e.g. to plot statistics 
            as in POS plot
        samples (bool, optional): 
            whether samples should be returned, e.g. as in word probability plot

    Returns:
        corrs: dictionary with correlations for Bert, EZ and BNC
        len_bin: length of individual bins
        tokens_normalized: if `tokens=True`, tokens are normalized and returned
        df_samples: 
            if samples=True, contains samples of each bin, otherwise empty 
            dataframe
    """
    if len(bins)==0:
        bins = sorted(df[bin][~pd.isnull(df[bin])].unique())

    corr_bert = []
    corr_roberta = []
    corr_t5 = []
    corr_ez = []
    corr_bnc = []
    len_bin = []

    df_samples = pd.DataFrame()

    if tokens:
        tokens_normalized = {}
        tokens_normalized['bert'] = {}
        tokens_normalized['roberta'] = {}
        tokens_normalized['t5'] = {}
        tokens_normalized['ez'] = {}
        tokens_normalized['bnc'] = {}
        tokens_normalized['tsr'] = {}

    for ind, bin_i in enumerate(bins):
        if bin=='pos' and not isinstance(bin, str):
            continue
        elif bin=='pos':
            tag = str(bins[ind])
        else:
            tag = bin_i


        if bin == 'cloze':
            values_min = sorted(df[bin].values)[np.min(bin_i)]
            values_max = sorted(df[bin].values)[np.max(bin_i)]
            df_tmp = df.query('@values_min<={}<@values_max and TT==TT'.format(bin))
        elif bin=='sen_len':
            values_min = bin_i[0]
            values_max = bin_i[1]
            df_tmp = df.query('@values_min<={}<@values_max'.format(bin))
        elif bin=='word_len':
            values_min = bin_i[0]
            values_max = bin_i[1]
            df_tmp = df.query('@values_min<={}<@values_max and TT==TT'.format(bin))
        else:
            df_tmp = df.query('{}==@tag'.format(bin))

        if tokens:
            tokens_normalized['bert'][tag] = df_tmp.bert_flow_11_normalized
            tokens_normalized['roberta'][tag] = df_tmp.roberta_flow_11_normalized
            tokens_normalized['t5'][tag] = df_tmp.t5_flow_11_normalized
            tokens_normalized['ez'][tag] = df_tmp.TT_normalized
            tokens_normalized['bnc'][tag] = df_tmp.x_bnc_normalized
            tokens_normalized['tsr'][tag] = df_tmp.x_normalized

        if samples:
            df_samples[str(ind)] = np.random.choice(df_tmp.words, 10)

        len_bin.append(len(df_tmp))
        tokens_tsr = []
        tokens_bert = []
        tokens_roberta = []
        tokens_t5 = []
        tokens_ez = []
        tokens_bnc = []
        for index, row in df_tmp.iterrows():
            if level=='sentence':
                tokens_tsr.extend(list(row['x'][1:-1]))
                tokens_bert.extend(list(row['bert_flow_11'][1:-1]))
                tokens_roberta.extend(list(row['roberta_flow_11'][1:-1]))
                tokens_t5.extend(list(row['t5_flow_11'][1:-1]))
                tokens_ez.extend(list(row['TT'][1:-1]))
                tokens_bnc.extend(list(row['x_bnc'][1:-1]))
            else:
                tokens_tsr.append(row['x'])
                tokens_bert.append(row['bert_flow_11'])
                tokens_roberta.append(row['roberta_flow_11'])
                tokens_t5.append(row['t5_flow_11'])
                tokens_ez.append(row['TT'])
                tokens_bnc.append(row['x_bnc'])

        corr_bert.append([tag, spearmanr(tokens_tsr, tokens_bert)[0]])
        corr_roberta.append([tag, spearmanr(tokens_tsr, tokens_roberta)[0]])
        corr_t5.append([tag, spearmanr(tokens_tsr, tokens_t5)[0]])
        corr_ez.append([tag, spearmanr(tokens_tsr, tokens_ez)[0]])
        corr_bnc.append([tag, spearmanr(tokens_tsr, tokens_bnc)[0]])

    corrs={}
    corrs['bert'] = corr_bert
    corrs['roberta'] = corr_roberta
    corrs['t5'] = corr_t5
    corrs['ez'] = corr_ez
    corrs['bnc'] = corr_bnc

    if tokens:
        return corrs, len_bin, tokens_normalized, df_samples
    else:
        return corrs, len_bin, df_samples
