from os.path import join
import matplotlib.pyplot as plt
from plotting.plotting import image_plot, line_plot, plot_input_reduction_pos_analysis, plot_ranking
from analysis.analysis import BinAnalysis
from _local_options import eval_folder
import pandas as pd
import click
import os
import numpy as np
import pickle
from proc.correlate import collect_ranking_over_seeds, average_dfs, include_rows
from general_utils import set_up_dir


@click.command()
@click.option("--rank", is_flag=True)
@click.option('--pos', is_flag=True)
@click.option("--labels", is_flag=True)
@click.option("--sen_len", is_flag=True)
@click.option("--word_len", is_flag=True)
@click.option("--word_prob", is_flag=True)
@click.option("--flip", is_flag=True)
def main(rank, pos, labels, sen_len, word_len, word_prob, flip): 
    # --rank --pos --labels --sen_len --word_len --word_prob --flip

    print('run analysis for ')
    print('rank') if rank else None
    print('pos') if pos else None
    print('labels') if labels else None
    print('sen_len') if sen_len else None
    print('word_len') if word_len else None
    print('word_prob') if word_prob else None
    print('flip') if flip else None

    ### CORRELATION RANKING ####
    if rank:
        plot_dir_ = os.path.join(eval_folder, 'correlation_ranking')
        set_up_dir(plot_dir_)

        df_template_dir_ = os.path.join(eval_folder, 'ignore_first_last')

        # If evaluation on multiple seeds
        # df_template_dir_ = 'experiments/ignore_first_last_seed_{}/'

        seeds = [1]
        for task in ['SR', 'TSR']:

            dfs_seeds = collect_ranking_over_seeds(df_template_dir_, task, seeds=seeds)

            dat_, df_, df_mean_token = average_dfs([dfs_seeds[i]['token'] for i in seeds])
            dat_, df_, df_mean_sent = average_dfs([dfs_seeds[i]['sentence'] for i in seeds])
            df_in = {'token': df_mean_token,
                     'sentence': df_mean_sent}

            pickle.dump(df_in, open(os.path.join(plot_dir_, 'corrs_{}.p'.format(task)), 'wb'))
            plot_ranking(df_in, plot_dir_,  task=task, include_rows = include_rows)

    analysis = BinAnalysis()
    labels_short = ['BERT', 'RoBERTa', 'T5', 'EZ', 'BNC']

    
    ### POS ####
    if pos:
        corr_pos_TSR, tokens_mean_TSR, labels_TSR, file_name = analysis.pos(analysis.df_TSR_words)
        corr_pos_SR, tokens_mean_SR, labels_SR, _ = analysis.pos(analysis.df_SR_words)

        image_plot([corr_pos_SR, corr_pos_TSR],
                   [tokens_mean_SR, tokens_mean_TSR],
                   labels_short,
                   [labels_SR, labels_TSR],
                   file_name)

        plt.subplots_adjust(wspace=0.05, hspace=0)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')

    ### LABELS ###
    if labels:
        from general_utils import label_dict_str, label_dict_sr
        corr_labels_SR, labels_SR = analysis.labels(analysis.df_SR, label_dict_sr)
        file_name = join(analysis.save_dir, analysis.prefix + 'SST_corr_labels_im')

        image_plot([corr_labels_SR],
                   [],
                   labels_short,
                   [labels_SR],
                   file_name,
                   tokens=False)

        corr_labels_TSR, labels_TSR = analysis.labels(analysis.df_TSR,label_dict_str)
        file_name = join(analysis.save_dir, analysis.prefix + 'wiki_corr_labels_im')

        image_plot([corr_labels_TSR],
                   [],
                   labels_short,
                   [labels_TSR],
                   file_name,
                   tokens=False)

    ### SEN_LEN ####
    if sen_len:

        bins_SR = [[1, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, 30], [31, 44]]
        bins_TSR = [[6, 10], [11, 15], [16, 20], [21, 25], [26, 30], [31, 50]]

        corr_sen_SR, file_name = analysis.sentence_length(analysis.df_SR, bins_SR)
        corr_sen_TSR, _ = analysis.sentence_length(analysis.df_TSR, bins_TSR)

        line_plot([corr_sen_SR, corr_sen_TSR], labels_short,
                  [bins_SR, bins_TSR], file_name, loc='upper right', rotation=30,
                  xlabel='sentence length', dashed=True)

    #### WORD_LEN ####
    if word_len:
        corr_word_SR, labels_SR, file_name = analysis.word_length(analysis.df_SR_words, task='SR')
        corr_word_TSR, labels_TSR, _ = analysis.word_length(analysis.df_TSR_words, task='TSR')

        line_plot([corr_word_SR, corr_word_TSR], labels_short, [labels_SR, labels_TSR], file_name, loc='lower right',
                  rotation=0, xlabel='word length', dashed=True)

    #### WORD_PROB ###
    if word_prob:
        corr_cloze_SR, labels, xticklabels_SR, samples_SR, file_name = analysis.word_probability(analysis.df_SR_words, task='SR')
        corr_cloze_TSR, labels_TSR, xticklabels_TSR, samples_TSR, _ = analysis.word_probability(analysis.df_TSR_words, task='TSR')

        line_plot([corr_cloze_SR, corr_cloze_TSR], labels, [xticklabels_SR, xticklabels_TSR], file_name,
                  samples=[samples_SR, samples_TSR], xlabel=r'word predictability $\times 10^{-3}$',
                  dashed=True, loc="lower right")


    ### INPUT REDUCTION ###
    if flip:
        
        out_dir_flip = os.path.join(eval_folder, 'input_reduction')

        if True:
            # output all_flip_cases_{}.p file from run_flipping.py

            sr_file = os.path.join(out_dir_flip, 'all_flip_cases_SR.p')
            tsr_file = os.path.join(out_dir_flip, 'all_flip_cases_TSR.p')

            for file_ in [sr_file, tsr_file]:
                if not os.path.isfile(file_):
                    print('Missing input reduction results file {}. Please run "run_flipping.py" first.'.format(file_))

        else:
            # paper dfs
            sr_file = 'data/all_flip_cases_SR.p'
            tsr_file = 'data/all_flip_cases_TSR.p'
        
        
        df_flips_sr = pd.read_pickle(sr_file)
        df_flips_tsr = pd.read_pickle(tsr_file)
        steps = len(df_flips_sr['generate']['tsr']['E'][0])
        fracs = np.linspace(0., 1., steps)

        model_cases = ['tsr', 
                       'ez_nr',
                       'base_bert_flow_11',
                       'base_roberta_flow_11',
                       'base_t5_flow_11',
                       'cnn0.50',
                       'sattn_rels_0.25',
                       'bnc_freq_prob',
                       'random'
                        ][::-1]

        results_dir = out_dir_flip
        plot_input_reduction_pos_analysis(df_flips_sr, df_flips_tsr, fracs, model_cases,  results_dir=results_dir)


if __name__ == '__main__':
    main()
