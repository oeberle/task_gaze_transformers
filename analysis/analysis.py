import pandas as pd
from os import makedirs
from os.path import join
import analysis.utils as utils
import numpy as np
from plotting.plotting import line_plot
from _local_options import eval_folder


class BinAnalysis:
    """
    Args:
        prefix (str, optional): ...
    """
    def __init__(self, prefix = 'base_'):
        """
        Initialize paths, prepare pd.DataFrame

        Args:
            prefix: 'base_' for Base-Bert and 'large_' for Large-Bert
        """
        self.save_dir = join(eval_folder, 'bin_analysis')
        self.prefix = prefix
        try:
            makedirs(self.save_dir)
        except FileExistsError:
            pass


        folder = eval_folder
        dfs_orig_TSR = pd.read_pickle(join(folder, 'ignore_first_last', 'dfs_all_TSR.p'))


        dfs_orig_SR = pd.read_pickle(join(folder, 'ignore_first_last', 'dfs_all_SR.p'))

        self.df_TSR, df_TSR_words = utils.prepare_dfs(dfs_orig_TSR)
        self.df_TSR_words = utils.standardize_columns(
            df_TSR_words, ['x', 'x_bnc', 'bert_flow_11', 'TT', 'roberta_flow_11', 't5_flow_11'])

        self.df_SR, df_SR_words = utils.prepare_dfs(dfs_orig_SR)
        self.df_SR_words = utils.standardize_columns(
            df_SR_words, ['x', 'x_bnc', 'bert_flow_11', 'TT', 'roberta_flow_11', 't5_flow_11'])

    def pos(self, df_words: pd.DataFrame):
        """
        Part-of-Speech Analysis

        Args:
            df_words (pd.DataFrame):
                dataframe on which POS tags are extracted and correlated with 
                fixations

        Returns:
            corr_pos: correlation values for Bert, EZ and BNC (one per row)
            tokens_mean: standardized token values
            labels: xticklabels for plot
            file_name: path to store plot
        """
        file_name = join(self.save_dir, self.prefix + 'corr_pos_im')
        
        # create additional column for POS in df
        df_words = utils.extract_pos(df_words)
        
        # correlate tokens wrt POS tag with human data
        corrs, len_pos, tokens, _ = utils.correlate_category(df_words, 'pos', 
                                                             samples=False)

        # extract the top-6 POS tags
        corrs_short = {}
        corrs_short['bert'] = [corrs['bert'][ipos] for ipos in 
                               np.argsort(len_pos)[::-1][:6].tolist()]
        corrs_short['roberta'] = [corrs['roberta'][ipos] for ipos in
                               np.argsort(len_pos)[::-1][:6].tolist()]
        corrs_short['t5'] = [corrs['t5'][ipos] for ipos in
                               np.argsort(len_pos)[::-1][:6].tolist()]
        corrs_short['ez'] = [corrs['ez'][ipos] for ipos in 
                             np.argsort(len_pos)[::-1][:6].tolist()]
        corrs_short['bnc'] = [corrs['bnc'][ipos] for ipos in 
                              np.argsort(len_pos)[::-1][:6].tolist()]

        # transform data into matrix
        corr_pos = np.zeros([5, 6])
        corr_pos[0] = np.array([corrs_short['bert'][i][1] for i in range(6)])
        corr_pos[1] = np.array([corrs_short['roberta'][i][1] for i in range(6)])
        corr_pos[2] = np.array([corrs_short['t5'][i][1] for i in range(6)])
        corr_pos[3] = np.array([corrs_short['ez'][i][1] for i in range(6)])
        corr_pos[4] = np.array([corrs_short['bnc'][i][1] for i in range(6)])

        # labels of selected POS tags
        labels = [corrs_short['bert'][i][0] for i in range(6)]

        tokens_mean = np.zeros([6, 6])

        # calculate average values for standardized tokens (i.e. importance 
        # scores) for lower part of plot
        for ipos, pos in enumerate(labels):
            tokens_mean[0, ipos] = np.mean(tokens['tsr'][pos])
            tokens_mean[1, ipos] = np.mean(tokens['bert'][pos])
            tokens_mean[2, ipos] = np.mean(tokens['roberta'][pos])
            tokens_mean[3, ipos] = np.mean(tokens['t5'][pos])
            tokens_mean[4, ipos] = np.mean(tokens['ez'][pos])
            tokens_mean[5, ipos] = np.mean(tokens['bnc'][pos])

        return corr_pos, tokens_mean, labels, file_name

    def labels(self, df, label_dict):
        """
        Correlation analysis based on class labels

        Args:
            df (pd.DataFrame):
                dataframe on which class labels are extracted and correlated 
                with fixations

        Returns:
            corr_labels: correlation values for Bert, EZ and BNC (one per row)
            labels: xticklabels for plot
        """
        corrs, len_pos, _ = utils.correlate_category(
            df, 'labels', level='sentence', tokens=False, samples=False)

        num_labels = len(df.labels.unique())

        corr_labels = np.zeros([5, num_labels])

        corr_labels[0] = np.array([corrs['bert'][i][1] for i in range(num_labels)])
        corr_labels[1] = np.array([corrs['roberta'][i][1] for i in range(num_labels)])
        corr_labels[2] = np.array([corrs['t5'][i][1] for i in range(num_labels)])
        corr_labels[3] = np.array([corrs['ez'][i][1] for i in range(num_labels)])
        corr_labels[4] = np.array([corrs['bnc'][i][1] for i in range(num_labels)])
        labels = label_dict.keys()

        return corr_labels, labels

    def sentence_length(self, df, bins):
        """
        Correlation analysis based on sentence length

        Args:
            df (pd.DataFrame): 
                dataframe on which sentence length is extracted and correlated 
                with fixations
            bins: 
                pre-defined bins for sentence length on which correlation is 
                calculated

        Returns:
            corr_sen: correlation values for Bert, EZ and BNC (one per row)
            file_name: path to store plot
        """
        df = utils.extract_sen_len(df)
        corrs, len_bin, _ = utils.correlate_category(
            df, 'sen_len', bins=bins, level='sentence', tokens=False, samples=False)

        corr_sen = np.zeros([5, len(corrs['bert'])])
        corr_sen[0] = np.array([corrs['bert'][i][1] for i in range(len(corrs['bert']))])
        corr_sen[1] = np.array([corrs['roberta'][i][1] for i in range(len(corrs['bert']))])
        corr_sen[2] = np.array([corrs['t5'][i][1] for i in range(len(corrs['bert']))])
        corr_sen[3] = np.array([corrs['ez'][i][1] for i in range(len(corrs['bert']))])
        corr_sen[4] = np.array([corrs['bnc'][i][1] for i in range(len(corrs['bert']))])

        file_name = join(self.save_dir, self.prefix + 'corr_sen_len')

        return corr_sen, file_name

    def word_length(self, df_words: pd.DataFrame, task: str):
        """
        Correlation analysis based on word length

        Args:
            df_words (pd.DataFrame): 
                dataframe on which word length is extracted and correlated 
                with fixations
            task (str): 
                "SR" or "TSR" for sentiment reading (SST) and task-specific 
                reading (Wikipedia)

        Returns:
            corr_word: correlation values for Bert, EZ and BNC (one per row)
            labels: xticklabels for plot
            file_name: path to store plot
        """
        df_words = utils.extract_word_len(df_words)

        if task=='SR':
            bins = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], 
                    [8, 9], [9, 10], [10, 11], [11, np.max(df_words['word_len'])]]
        else:
            bins = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], 
                    [8, 9], [9, 10], [10, np.max(df_words['word_len'])]]

        corrs, len_bin, _ = utils.correlate_category(
            df_words, 'word_len', bins=bins, level='word', tokens=False, 
            samples=False)

        corr_word = np.zeros([5, len(corrs['bert'])])
        corr_word[0] = np.array([corrs['bert'][i][1] for i in range(len(corrs['bert']))])
        corr_word[1] = np.array([corrs['roberta'][i][1] for i in range(len(corrs['bert']))])
        corr_word[2] = np.array([corrs['t5'][i][1] for i in range(len(corrs['bert']))])
        corr_word[3] = np.array([corrs['ez'][i][1] for i in range(len(corrs['bert']))])
        corr_word[4] = np.array([corrs['bnc'][i][1] for i in range(len(corrs['bert']))])

        xlabels = [bins[i][0] for i in range(len(bins))]
        xlabels[-1] = '>{}'.format(xlabels[-2])

        file_name = join(self.save_dir, self.prefix + 'corr_word_len')

        return corr_word, xlabels, file_name

    def word_probability(self, df_words: pd.DataFrame, task: str):
        """
        Correlation analysis based on word probability. .
        Results are plotted within this function, so nothing will be returned.

        Args:
            df_words (pd.DataFrame): 
                dataframe on which word probability is extracted and correlated 
                with fixations
            task (str): 
                "SR" or "TSR" for sentiment reading (SST) and task-specific 
                reading (Wikipedia)
        """
        nbins = 13
        len_bins = int(np.ceil(len(df_words) / nbins))
        bins = [[i * len_bins, (i + 1) * len_bins] for i in range(nbins)]
        bins[-1][1] = len(df_words) - 1

        xticklabels = ["{:.3f}".format(1000 * sorted(df_words.cloze.values)[i1])
                       if 1000 * sorted(df_words.cloze.values)[i1] < 0.01
                       else
                       "{:.2f}".format(1000 * sorted(df_words.cloze.values)[i1])
                       if 0.001 < 1000 * sorted(df_words.cloze.values)[i1] < 0.1
                       else
                       "{:.1f}".format(1000 * sorted(df_words.cloze.values)[i1])
                       if 0.01 < 1000 * sorted(df_words.cloze.values)[i1] < 1
                       else
                       int(np.round(1000 * sorted(df_words.cloze.values)[i1]))
                       for i0, i1 in bins]

        corrs, size_bin, df_samples = utils.correlate_category(
            df_words, 'cloze', bins=bins, level='word', tokens=False, 
            samples=True)

        corr_cloze = np.zeros([5, len(corrs['bert'])])
        corr_cloze[0] = np.array([corrs['bert'][i][1] for i in range(len(corrs['bert']))])
        corr_cloze[1] = np.array([corrs['roberta'][i][1] for i in range(len(corrs['roberta']))])
        corr_cloze[2] = np.array([corrs['t5'][i][1] for i in range(len(corrs['t5']))])
        corr_cloze[3] = np.array([corrs['ez'][i][1] for i in range(len(corrs['bert']))])
        corr_cloze[4] = np.array([corrs['bnc'][i][1] for i in range(len(corrs['bert']))])

        labels = ['BERT', 'RoBERTa', 'T5', 'EZ', 'BNC']
        file_name = join(self.save_dir, self.prefix + 'corr_cloze')

        delimiter = '\n'

        samples = [
            df_samples.loc[0, str(i)] + '\n' +
            df_samples.loc[1, str(i)] + '\n' +
            df_samples.loc[2, str(i)] + '\n' +
            df_samples.loc[3, str(i)] + '\n' +
            df_samples.loc[4, str(i)] 
            if i % 2 == 0 else '' for i in range(len(xticklabels))
        ]

        samples = [delimiter.join(sorted(samples[i].split('\n'), key=len)) if i % 4 == 0
                      else delimiter.join(sorted(samples[i].split('\n'), key=len, reverse=True))
                      for i in range(len(samples))]

        return corr_cloze, labels, xticklabels, samples, file_name
