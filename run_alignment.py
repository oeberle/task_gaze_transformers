import pandas as pd
import pickle
import yaml
from os import path
from proc.compare import SentenceAlignment
from eval_utils import *
from general_utils import set_up_dir
import click
from dataloader.data_loader_zuco import ZucoLoader
from _local_options import eval_folder, model_output, zuco_files

@click.command()
@click.option('--analysis_file')
@click.option('--task')
@click.option('--filter_tokens', default='ignore_first_last')

def main(analysis_file, task, filter_tokens):

    """
    Main analysis function to compute correlation dataframes and the flipping experiments

    Args:
        analysis_file (str): Filename of the analysis yaml, see configs/analysis/*.yaml
        task (str): {"TSR", "NR", "SR"}
        filter_tokens  (str): {"ignore_first_last", None}, used to mask attribution vectors

    Returns:
        None
    """
    analysis_dir = path.abspath('configs/analysis')
    if analysis_file is None:
        if task == 'TSR':
            analysis_file = 'wikirel_base_pub.yaml'
        elif task == 'SR':
            analysis_file = 'sst_base_pub.yaml'
        else:
            raise NotImplementedError

    with open(path.join(analysis_dir, analysis_file), 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    cfg = {k: [v[0].format(model_output), v[1]] for k, v in cfg.items()}

    if task == 'TSR' and 'nr' in cfg:
        duplicates_flag = True
    else:
        duplicates_flag = False
    cases = [(k, v) for k, v in cfg.items() if k != 'exclude_from_corr']

    out_dir = path.join(eval_folder, filter_tokens) \
        if filter_tokens == 'ignore_first_last' \
        else path.join(eval_folder, 'ignore_none')

    set_up_dir(out_dir)
    print('Results will be stored in:', out_dir)

    # Load base df
    ZM = ZucoLoader(zuco1_prepr_path=zuco_files, zuco2_prepr_path=None)
    df_human_tsr = ZM.get_zuco_task_df(zuco_version=1, task=task)
    df_human_avg_tsr = get_human_avg_df(df_human_tsr, ignore_zeros=False)

    if task == 'TSR':
        df_human_nr = ZM.get_zuco_task_df(zuco_version=1, task='NR')
        df_human_avg_nr = get_human_avg_df(df_human_nr,  ignore_zeros=False)

    if duplicates_flag:
        duplicates = list(set(df_human_avg_tsr.text_id) & set(df_human_avg_nr.text_id))
        # Set zuco dfs
        df_nr = df_human_avg_nr.query('text_id in @duplicates').set_index('text_id')
        df_tsr = df_human_avg_tsr.query('text_id in @duplicates').drop_duplicates(
            subset='text_id', keep='last').set_index('text_id')
    else:
        df_nr = df_human_avg_nr.set_index('text_id') if task == 'TSR' else None
        df_tsr = df_human_avg_tsr.drop_duplicates(subset='text_id', keep='last').set_index('text_id')

    dfs_all = {}
    procs = {'abs': proc_rel_abs}
    cases_to_filter = []
    ref_ids = []

    match_threshold = 14 if task in ['TSR', 'NR'] else 22

    # Collect all dfs and align order of sentences
    allowed_comps = {}
    for case, (eval_file, x_comp) in cases:

        if case in ['tsr', 'nr'] or case.startswith('ez_nr'):
            continue

        print(case, eval_file, '\n')

        try:
            df_ = pd.read_pickle(eval_file)
        except FileNotFoundError:
            print('Data frame with model attributions could not be found, please download the files (see README).')

        df_.index = df_.index.rename('sample_id')
        remove_list, include_list, extra_mapping, allowed_matches = get_matching_vars(case, task)

        if 'bert_flow' in case or 'roberta_flow' in case or 't5_flow' in case:
            x_rep = 'x' + '_flow_' + case.split('_')[-1]
        elif 'bert_rollout' in case:
            x_rep = 'x' + '_rollout_' + case.split('_')[-1]
        elif 'bnc_freq' in case:
            x_rep = x_comp
        elif 'bert_x' in case or 'bert_max' in case:
            x_rep = x_comp
        elif 'roberta_x' in case or 'roberta_max' in case:
            x_rep = x_comp
        elif 't5_x' in case or 't5_max' in case:
            x_rep = x_comp
        elif '_saliency' in case:
            x_rep = x_comp
        elif '_last' in case:
            x_rep = 'x_last'
        else:
            x_rep = 'x'

        print("Align sentences")
        E = SentenceAlignment(df_tsr,
                              x=x_rep,
                              match_threshold=match_threshold,
                              remove_list=remove_list,
                              include_list=include_list,
                              allowed_matches=allowed_matches,
                              extra_mapping=extra_mapping
                              )

        _, idx_mapping = E.compare_df_to_human(df_)


        ids = list(map(list,zip(*idx_mapping)))
        # Extract aligned eval dfs
        df1 = E.df_proc.loc[ids[0]]

        df1['text_id'] = ids[1]
        df1 = df1.set_index('text_id')

        if 'relevances' in eval_file.split('/')[-1]:
            df1, new_cols = add_proc_funcs(df1, procs)
            cases_to_filter.append([case, 'x'])

        allowed_comps[case] = x_comp

        dfs_all[case] = df1
        ref_ids.append(ids[1])

    if 'tsr' in cfg:
        dfs_all['tsr'] = df_tsr.loc[ids[1]]
        allowed_comps['tsr'] = 'x'

    if 'nr' in cfg and duplicates_flag:
        dfs_all['nr'] = df_nr.loc[ids[1]]
        allowed_comps['nr'] = 'x'

    if 'ez_nr' in cfg:
        all_ez_nr = [c for c in cfg if c.startswith('ez_nr')]
        all_ez_nr_dict = {}
        for i,case_ez in enumerate(all_ez_nr):
            eval_file, x_comp =  cfg[case_ez]

            df_ez = pd.read_pickle(eval_file)
            df_ez = df_ez.set_index('text_id')
            df_ez = df_ez.rename(columns={'word': 'words'})

            df_ez[x_comp] = df_ez[x_comp].apply(lambda x: np.array(x))
            allowed_comps[case_ez] = x_comp

            idx_mapping, remove_ids_df_ref, remove_ids_df = E.proc_ez_matching(df_ez, df1)

            df_ez = df_ez[~df_ez.index.isin(remove_ids_df)]

            ids_ez = list(map(list,zip(*idx_mapping)))
            all_ez_nr_dict[case_ez] = df_ez

            if len(remove_ids_df_ref) > 0:
                dfs_all_ = {}
                for k, df_ in dfs_all.items():
                    if k in all_ez_nr+['tsr']:
                        #Adding all to dfs_all
                        for ez_case in all_ez_nr:
                            dfs_all_[ez_case] = all_ez_nr_dict[ez_case].loc[ids_ez[0]]
                        dfs_all_['tsr'] = dfs_all['tsr'].loc[ids_ez[0]]
                    else:
                        # sample_ids
                        dfs_all_[k] = dfs_all[k].loc[ids_ez[1]]

                dfs_all = dfs_all_
            else:
                for ez_case in all_ez_nr_dict.keys():
                    if ez_case not in dfs_all:
                        dfs_all[ez_case] = all_ez_nr_dict[ez_case].loc[ids_ez[0]] #df_ez.loc[ids_ez[0]]

    if task in ['TSR', 'NR']:
        dfs_all = {k: filter_duplicates(df_) for k, df_ in dfs_all.items()}

    pickle.dump(dfs_all, open(path.join(out_dir, 'dfs_all_{}.p'.format(task)), 'wb'))


if __name__ == '__main__':
    main()