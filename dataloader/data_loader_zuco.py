import numpy as np
import pandas as pd
import os
import h5py
import warnings
import difflib
from glob import glob

from tqdm import tqdm

from typing import List, Tuple, Union, Optional

import dataloader.utils.etc as uetc
import dataloader.utils.io as uio
import dataloader.utils.pandas as upd
from general_utils import all_subjects, all_subjects2



def load_all_tables_from_preprocessed(zuco1_prepr_path: str,
                                      zuco2_prepr_path: str
                                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads all three tables (texts_table, words_table, and fixations_table) 
    for both Zuco 1 and 2 from `preprocessed` dirs and returns them, 
    concatenated.

    Args:
        zuco1_prepr_path (str):
            Directory with pickle files `texts_table.p`, `words_table.p`, and 
            `fixations_table.p` for Zuco 1.
        zuco2_prepr_path (str): 
            Directory with pickle files `texts_table.p`, `words_table.p`, and 
            `fixations_table.p` for Zuco 2.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            texts_table, words_table, fixations_table
    """
    pickle_files_zuco1 = glob(os.path.join(zuco1_prepr_path, "*.p"), 
                              recursive=True)
    pickle_files_zuco2 = glob(os.path.join(zuco2_prepr_path, "*.p"), 
                              recursive=True)
    zuco_versions = [1]*len(pickle_files_zuco1) + [2]*len(pickle_files_zuco2)
    dfs = {}

    for pf, zv in zip(pickle_files_zuco1 + pickle_files_zuco2, zuco_versions):
        _, fname = os.path.split(pf)
        fname    = fname[:-2] # without file extension
        df       = pd.read_pickle(pf)
        if fname != "texts_table":
            df.loc[:, "_zuco_version"] = zv
        dfs[fname, zv] = df

    for table_name in ["texts_table", "words_table", "fixations_table"]:
        for zv in [1,2]: # zuco version
            assert (table_name, zv) in dfs.keys(), \
                f"{table_name} for zuco version {zv} missing"
    assert dfs["texts_table",1].equals(dfs["texts_table",2]), \
        "texts_tables aren't equal"
    assert "_zuco_version" in dfs["texts_table",1].columns, \
        "texts_tables do not appear to be for both Zuco1 and 2 in one df"

    tt = dfs["texts_table", 1]
    wt = pd.concat([dfs["words_table",1], dfs["words_table",2]], axis=0)
    ft = pd.concat([dfs["fixations_table",1], dfs["fixations_table",2]], axis=0)

    return tt, wt, ft

def fixations_table_from_mat(mat_path: str, 
                             min_fix_duration: float=100,
                             text_id_lookup: Optional["TextIDLookup"]=None
                            ) -> pd.DataFrame:
    """
    Generates the "fixations"-table from ZuCo-data.
    
    Args:
        mat_path (str): 
            path to a single matlab (.mat) file of the ZuCo data
        min_fix_duration (float): 
            Minimum duration of a single fixation in ms. Fixations
            that are shorter will be dropped.
        text_id_lookup (TextIDLookup or None):
            instance of TextIDLookup
        boil_down_pattern: (str)
            See TextIDLookup

    Columns:
        fix_pos (int64): index (zero-based) of fixation in this reading
        x (float64): x-coordinate of fixation
        y (float64): y-coordinate of fixation
        word_pos (Int64): position (zero-based) of word in corresponding text
        duration (float64): duration of fixation in ms
        _i_text_in_mat (int32): index of text in .mat file
        text_id (int64): 
            text_id as assigned by `text_id_lookup`. All None, if 
            `text_id_lookup` is None.
    """
    with h5py.File(mat_path, mode="r") as mf:
        n_texts  = _number_of_texts(mf)
        granules = []
        for i_text in range(n_texts):
            fix_tab_granule = _fixations_table_granule_from_mat(
                mf, i_text, min_fix_duration=min_fix_duration
            )
            
            if fix_tab_granule.size > 0: # if there are any fixations
                # tag each row of granule with i_text
                fix_tab_granule.loc[:, "_i_text_in_mat"] = i_text
                
                # determine (global) text_id
                if text_id_lookup is None:
                    text_id = None
                else:
                    text = _get_from_sentenceData(
                        mf, "content", i_text, is_str=True)
                    text_id, is_exact_match = text_id_lookup.lookup(text)
                fix_tab_granule.loc[:, "text_id"] = text_id

                granules.append(fix_tab_granule)
            
    # concat all fixation tables
    fixations_table = pd.concat(granules, ignore_index=True)
    fixations_table = fixations_table.astype({"_i_text_in_mat":"int32"})
    
    return fixations_table

def fixations_table_from_many_mats(mat_paths: List[str],
                                   min_fix_duration:float=100,
                                   text_id_lookup : Optional["TextIDLookup"]=None,
                                   subj_task_id_from_mat_path = "default",
                                   verbose:bool = True):
    """
    Wrapper for function fixations_table_from_mat to load several matlab files 
    from ZuCo at once.
    
    Args:
        verbose: 
            Whether or not to print progress
        subj_task_id_from_mat_path:
            Either...
                "default", which assumes the original filenames like 
                "resultsABC_DEF" for which it would infer the subject_id 'ABC' 
                and task_id 'DEF'
            or...                    
                Some function with signature f(mp:str) -> (str,str) 
                where the input (mp) is a .mat-filepath and the output consists 
                of subject_id and task_id that are to be used as values for the 
                respective columns in the final fixations_table.
        text_id_lookup (TextIDLookup or None):
            instance of TextIDLookup
            
    Returns:
        pandas.DataFrame with columns:
            columns as described in `fixations_table_from_mat` (see doc),
            furthermore: subject_id (str), e.g. "YFR", and task_id 
            (str: {"NR", "TSR", "SR"})
    """
    fix_tables = [] # one per mat-file, thus one per subject and task
    for i_file, mp in enumerate(tqdm(mat_paths, disable=not verbose, 
                                     desc="making fixations table")):
        ftable = fixations_table_from_mat(
            mp, min_fix_duration = min_fix_duration,
            text_id_lookup = text_id_lookup)
        
        if subj_task_id_from_mat_path == "default":
            subj_task_id_from_mat_path = _subject_and_task_id_from_filename
        
        subject_id, task_id = subj_task_id_from_mat_path(mp)
        ftable.loc[:, "subject_id"] = subject_id
        ftable.loc[:, "task_id"]    = task_id
        
        fix_tables.append(ftable)
            
    return pd.concat(fix_tables, ignore_index=True)

def append_labels_to_fixations(fix_table: pd.DataFrame, zuco_version: int,
                               texts_table: pd.DataFrame, 
                               task_material_tsr_path: Optional[str]=None
                              ) -> pd.DataFrame:
    """
    Appends a column to a fixations table containing the label of the respective
    fixation. 

    Parameters
    ----------
    fix_table : pd.DataFrame
        fixations table. Will be copied.
    zuco_version : int : {1,2}
        The zuco version for which `fix_table` was generated. 
    texts_table : pd.DataFrame
        Texts table containing a column `_zuco_version`, thus generated for 
        both Zuco 1 and 2 together. (see texts_table_from_task_material)
    task_material_tsr_path : Optional[str], optional
        Path to `relations_labels_task3.csv` of Zuco 1 task material. If 
        `_zuco_version` is 1, this should not be None since some labels can only
        be filled in with this task material at hand. By default None

    Returns
    -------
    pd.DataFrame
        Copy of `fix_table` with an additional column `label` (str)

    Raises
    ------
    ValueError
        Raised if `texts_table` does not have any column `_zuco_version`.
    """
    if "_zuco_version" not in texts_table.columns:
        raise ValueError("texts_table must have a column _zuco_version")

    fix_table = fix_table.copy() # avoid side effects
    texts_table = texts_table.query("_zuco_version == @zuco_version")

    if "label" in fix_table.columns:
        warnings.warn("label column already in fix_table. Will be overwritten.")
        fix_table.drop(columns="label", inplace=True)

    if task_material_tsr_path is None:
        if zuco_version == 1:
            warnings.warn("Not all fixations will have a label since zuco "
                          "version was set to 1 whereas task_material_tsr_path "
                          "is None.")
    else:
        if zuco_version == 2:
            warnings.warn("task_material_tsr_path is ignored since "
                          "zuco_version == 2.")
        # get labels for all fixations from task TSR in Zuco 1 ready
        tm_tsr1 = _load_task_material_tsr(task_material_tsr_path)

        # lut: look-up table
        label_lut_tsr1 = (
            tm_tsr1
            [["relation_type"]]
            .reset_index() # order of texts in csv and in mat files is the same
            .rename(columns={"index": "_i_text_in_mat", # as in fix_table
                             "relation_type": "label_tsr1"})
            .assign(task_id = "TSR")
        )

        # join labels
        fix_table = fix_table.merge(
            label_lut_tsr1, on=["_i_text_in_mat","task_id"], how="left")
    
    # get labels for all fixations NOT   from task TSR in Zuco 1   ready
    is_zuco_1 = texts_table["_zuco_version"] == 1
    is_tsr    = texts_table["task_id"] == "TSR"

    label_lut_other = (
        texts_table
        [~(is_zuco_1 & is_tsr)] # zuco 1 TSR covered by label_lut_tsr1
        [["text_id","task_id","_labels"]]
        .explode("_labels")
        .query("_labels.notna()")
    )

    # join labels
    fix_table = fix_table.merge(
        label_lut_other, on=["text_id","task_id"], how="left")

    if task_material_tsr_path is not None:
        fix_table.loc[:, "label"] = (
            fix_table["label_tsr1"]
            .where(fix_table["label_tsr1"].notna(), other=fix_table["_labels"]))
    else:
        fix_table.rename(columns={"_labels": "label"}, inplace=True)
    
    fix_table.drop(columns=["_labels","label_tsr1"], errors="ignore", 
                   inplace=True) 
    # ignore errors: "label_tsr1" will be missing if no task material path passed
    
    return fix_table

def fill_gaps_in_fixations(fix_table: pd.DataFrame, words_table: pd.DataFrame,
                           duration_0: bool=True, 
                           append_word_column: bool=False) -> pd.DataFrame:
    """
    Inserts rows for all words that were not fixated into a fixation table.

    Args:
        fix_table (pd.DataFrame): 
            fixation table. Must have column `_zuco_version`.
        words_table (pd.DataFrame): 
            words table. Must have column `_zuco_version`.
        duration_0 (bool, optional): 
            If True, NaNs in the column `duration` will be replaced by 0. 
            Defaults to True.
        append_word_column (bool, optional): 
            If True, the column `word` will be joined to `fix_table`. 
            Defaults to False.
            This functionality doesn't fit the function name but since it's
            so easy to do it altogether... 

    Returns:
        pd.DataFrame: fixations_table
    """
    fix_table   = fix_table.copy() # avoid side effects
    words_table = words_table.drop_duplicates(
        subset=["word_pos","text_id","_zuco_version"])

    # replace NaN in word_pos by -1, since merging would fail otherwise (later)
    fix_table.loc[:, "word_pos"] = fix_table["word_pos"].fillna(-1)

    # df with columns: _zuco_version, text_id, word_pos
    all_word_pos = (
        words_table
        [["_zuco_version","text_id","word_pos"]
         + (["word"] if append_word_column else []) ]
        .astype({"word_pos": float})
    )

    # df with columnns subject_id, task_id, text_id, _zuco_version, 
    #                  _i_text_in_mat, label, word_pos
    fix_index_with_complete_word_pos = (
        fix_table
        [["subject_id","task_id","text_id","label","_zuco_version",
          "_i_text_in_mat"]]
        .drop_duplicates()
        .merge(all_word_pos, on=["_zuco_version","text_id"], how="left")
    )
    
    join_cols = [col for col in fix_index_with_complete_word_pos.columns
                 if col != "word"] # in case append_word_column=True
    fix_table = (
        fix_table
        .astype({"word_pos": float}) # Int64 is not really join-friendly
        .merge(fix_index_with_complete_word_pos, 
               on  = join_cols, 
               how = "outer")
        .astype({"word_pos": "Int64"})
    )

    # recover NaN in word_pos, which was replaced by -1 earlier
    fix_table.loc[fix_table["word_pos"] == -1, "word_pos"] = np.nan

    if duration_0:
        fix_table.loc[:,"duration"] = fix_table["duration"].fillna(0)

    return fix_table

def make_word_trt_tuple_table(fix_table: pd.DataFrame, 
                              words_table: pd.DataFrame, 
                              nan_replacement: str="NA"
                             ) -> pd.DataFrame:
    """
    Transforms fixations_table and words_table into a dataframe that contains 
    in addition to its index two columns: `word` and `duration`. 
    These contain tuples of the same length (same length per row), where the 
    i-th elements of two corresponding tuples are the fixated word (str) and 
    the total duration (float) the respective subject fixated this word (over 
    all fixations). 

    Args:
        fix_table (pd.DataFrame): 
            fixations table as generated by `fixations_table_from_many_mats`
        words_table (pd.DataFrame): 
            words table as generated by `words_table_from_many_mats`
        nan_replacement (str, optional): 
            string to use to replace NaNs in column "label". Defaults to "NA".
            Required as otherwise rows with NaN as label will be dropped.

    Returns:
        pd.DataFrame: 
            index columns:
                `subject_id`, `task_id`, `text_id`, `label`, `_zuco_version`
            value colums:
                `word` (Tuple[str]), `duration` (Tuple[float])
    """    


    """
    Transforms fixations_table and words_table into a dataframe that contains 
    in addition to its index two columns: `word` and `duration`. 
    These contain tuples of the same length (same length per row), where the 
    i-th elements of two corresponding tuples are the fixated word (str) and 
    the total duration (float) the respective subject fixated this word (over 
    all fixations). 

    Parameters
    ----------
    fix_table : pd.DataFrame
        Fixations table.
    words_table : pd.DataFrame
        Words table.

    Returns
    -------
    pd.DataFrame
        with index:
            `subject_id`, `task_id`, `text_id`, `label`, `_zuco_version`
        with cols:
            word : Tuple[str]
            duration : Tuple[float]
    """
    fix_words = fill_gaps_in_fixations(
        fix_table, words_table, append_word_column=True)

    main_index = ["subject_id","task_id","text_id","label","_zuco_version",
                  "_i_text_in_mat"]
    
    # sanity check
    nunique_words = (
        fix_words
        [main_index+["word","word_pos"]]
        .groupby(main_index+["word_pos"])
        .agg({"word":"nunique"})
        ["word"]
    )
    assert (nunique_words == 1).all(), \
        "A word_pos in some text has more than one word assigned to it"

    tuple_table = (
        fix_words
        .sort_values("fix_pos")
        [main_index + ["word_pos","word","duration"]]

        # can't group on NaNs (they will be dropped silently, thanks for 
        # nothing), so replace them by `nan_replacement` (default "NA")
        .fillna({"label": nan_replacement})

        .groupby(main_index + ["word_pos","word"]) 
        .sum() # only duration is aggregated
        .reset_index()
        
        # now put words and durations into lists (actually tuples since hashable)
        .sort_values("word_pos")
        .drop(columns="word_pos")
        .groupby(main_index)
        .agg(tuple)
    )

    return tuple_table

def texts_table_from_task_material(zuco_version: Union[int,str], *args, **kwargs
                                  ) -> pd.DataFrame:
    """
    Wrapper for all the `texts_table` functions for Zuco.

    Args:
        zuco_version (int or str): one of {1, 2, 'both'}

    Details:
        `text_id` will start at 2000 for ZuCo2 task material.

    Returns:
        pandas.DataFrame, columns: 
            see doc of `_texts_table_from_task_material_1_and_2`
    """
    if zuco_version == "both":
        return _texts_table_from_task_material_1_and_2(*args, **kwargs)
    elif zuco_version == 1:
        return _texts_table_from_task_material_zuco_1(*args, **kwargs)
    elif zuco_version == 2:
        return _texts_table_from_task_material_zuco_2(*args, **kwargs)
    else:
        raise ValueError("zuco_version must be 1, 2, or 'both'")

def _texts_table_from_task_material_1_and_2(task_material_1_dir: str, 
                                            task_material_2_dir: str
                                           ) -> pd.DataFrame:
    """
    Generates the texts_table for both zuco1 and -2 task-material. Zuco2 texts
    which occur in Zuco1 as well ('overlap'), get the same text_id as the Zuco1-
    counterpart. All other Zuco2-text_ids start from 2000.

    Args:
        task_material_1_dir (str): 
            Directory containing the csv-files 
                `sentiment_labels_task1.csv`
                `relations_labels_task2.csv`
                `relations_labels_task3.csv`
                `duplicate_sentences.csv`
            as supplied by the Zuco authors for Zuco_1.
        task_material_2_dir (str): 

    Returns:
        pandas.DataFrame with columns (incomplete; arbitrary order):
            text_id (int): identifier for each text
            text (str)
            task_id (str)
            length (int): number of characters in corresponding text
            _labels (tuple of str): 
                ... Hint: use pd.DataFrame.explode() to 'stretch' out to several
                rows.
            _zuco_version (int): 
                1 or 2, depending on the source task-material from where the 
                respective row was loaded.
            _overlap (bool): 
                True, if the respective text occurs in both versions of Zuco. 
            _is_control (bool): 
                True, if the respective text was labeled as control sentence
                in the respective task material.
    """
    # load text tables for zuco 1 and 2
    texts_table_1 = _texts_table_from_task_material_zuco_1(
        csv_path_sr  = os.path.join(task_material_1_dir, "sentiment_labels_task1.csv"),
        csv_path_nr  = os.path.join(task_material_1_dir, "relations_labels_task2.csv"),
        csv_path_tsr = os.path.join(task_material_1_dir, "relations_labels_task3.csv"),
        csv_path_ds  = os.path.join(task_material_1_dir, "duplicate_sentences.csv")
    ).assign(_zuco_version = 1)

    texts_table_2 = _texts_table_from_task_material_zuco_2(
        task_material_dir = task_material_2_dir,
        return_additional_columns = True, 
        verbose = False
    ).assign(_zuco_version = 2)
    
    # lookup text_ids of texts in zuco1 which exist in zuco2 as well
    tid_lookup = TextIDLookup(texts_table_1)
    new_text_id = []
    exact_match = []
    overlap     = []
    for _, row in texts_table_2.iterrows():
        lookup_result = tid_lookup.lookup(row["text"])
        if lookup_result is None: # no overlap, keep zuco_2 text_id
            overlap.append(False)
            new_text_id.append(row["text_id"])
            exact_match.append(False)
        else: # text exists in zuco_1 and zuco_2; pick zuco_1 text_id
            overlap.append(True)
            new_text_id.append(lookup_result[0])
            exact_match.append(lookup_result[1])

    texts_table_2 = texts_table_2.assign(
        _old_text_id = texts_table_2["text_id"],
        text_id      = new_text_id,
        _overlap     = overlap,
        _exact_match_with_zuco1 = exact_match
    )
    
    # tag text_ids in zuco1 which occur in zuco2 as well
    texts_table_1.loc[:, "_overlap"] = texts_table_1["text_id"].isin(
        texts_table_2["text_id"].values)
    
    # make _labels column in zuco2
    texts_table_2.rename(columns={"_tsr_relation_type": "_labels"}, 
                         inplace=True)
    texts_table_2 = upd.implode(texts_table_2, implode_cols="_labels")
    # replace (nan,) by ()
    texts_table_2.loc[:,"_labels"] = [() if pd.isnull(tpl[0]) else tpl 
                                      for tpl in texts_table_2["_labels"]]
    
    # concatenate both texts_tables and choose subset of columns
    texts_table_2 = texts_table_2[
        ["text_id","text","task_id","_sentence_id","_paragraph_id","_labels",
         "_is_control","length","_zuco_version","_overlap"]]
    texts_table = pd.concat([texts_table_1, texts_table_2], 
                            ignore_index=False, axis=0)
    
    texts_table = (
        texts_table
        .sort_values(["text_id","task_id","_zuco_version"])
        .reset_index(drop=True)
    )
    return texts_table

def _texts_table_from_task_material_zuco_2(task_material_dir: str, 
                                           return_additional_columns: bool=False, 
                                           verbose: bool=True) -> pd.DataFrame:
    # load material for task NR ('nr_1.csv' ... 'nr_7.csv')
    nr_dfs = []
    for i in range(1,8):
        fname       = f"nr_{i}.csv"
        nr_csv_path = os.path.join(task_material_dir, fname)
        if verbose:
            print("READING", nr_csv_path)
        nr_dfs.append(
            pd.read_csv(
                nr_csv_path, header=None, delimiter=";",
                names=["_sentence_id","_paragraph_id","text","_nr_control"])
            .assign(_filename = fname))
    nr_texts = pd.concat(nr_dfs).assign(task_id = "NR")
    
    # load material for task TSR ('tsr_1.csv' ... 'tsr_7.csv')
    tsr_dfs = []
    for i in range(1,8):
        fname        = f"tsr_{i}.csv"
        tsr_csv_path = os.path.join(task_material_dir, fname)
        if verbose:
            print("READING", tsr_csv_path)
        tsr_dfs.append(
            pd.read_csv(
                tsr_csv_path, header=None, delimiter=";",
                names=["_sentence_id","_paragraph_id","text",
                       "_tsr_relation_type"])
            .assign(_filename = fname))
    tsr_texts = pd.concat(tsr_dfs).assign(task_id = "TSR")
    
    # combine loaded data of NR and TSR
    texts_table = (
        pd.concat([nr_texts, tsr_texts], ignore_index=True, sort=True)
        .sort_values(["task_id","_sentence_id","_paragraph_id"])
    )
    
    text_id_table = (
        texts_table
        [["text"]]
        .drop_duplicates()
        .assign(text_id = lambda df: range(2000, df.shape[0]+2000))
    )
    
    # join text ids to texts table
    texts_table = texts_table.merge(
        text_id_table, on="text", how="left"
    )
    
    # determine lengths of texts
    texts_table.loc[:, "length"] = texts_table.text.apply(
        lambda t: len(t.split())
    )
    

    # for convenience
    texts_table.loc[:,"_is_control"] = (
        (texts_table["_nr_control"] == "CONTROL") 
        | (texts_table["_tsr_relation_type"] == "CONTROL")
    )
    
    #
    columns = ["text_id","text","task_id","length","_sentence_id","_paragraph_id"]
    if return_additional_columns:
        columns += ["_tsr_relation_type", "_is_control", "_filename"]
    return texts_table[ columns ]

def _texts_table_from_task_material_zuco_1(csv_path_sr: str,
                                           csv_path_nr: str,
                                           csv_path_tsr: str,
                                           csv_path_ds: str
                                          ) -> pd.DataFrame:
    """
    Generates the texts table from the task material
    provided by the ZuCo authors along with the results-mat-
    files. 
    
    All filenames listed below are the original ones when 
    downloaded here: https://osf.io/q3zws/files/
    
    Args:
        csv_path_sr (str): 
            path of the file that is originally named
            "sentiment_labels_task1.csv"
        csv_path_nr (str): 
            path of the file that is originally named
            "relations_labels_task2.csv"
        csv_path_tsr (str): 
            path of the file that is originally named
            "relations_labels_task3.csv"
        csv_path_ds (str):
            path of the file that is originally named
            "duplicate_sentences.csv"
            
    Returns:
        pandas.DataFrame with columns:
            text_id (int): unique for text
            text (str):
            task_id (str): one of "SR", "TSR", and "NR"
            _sentence_id (int): as assigned by ZuCo authors
            _paragraph_id (float): as assigned by ZuCo authors; 
                float only because it contains NaNs 
                (all rows in task "SR")
            _label (list): Each cell contains a list of all labels assigned to
                this text_id for this task, e.g. ["EMPLOYER","VISITED"].
            length (int): length of text obtained by 
                text.split()
                
        * loosely match texts, such that text-id refers to same
        content even if the text is not bitwise equal
        Even in the task material, some sentences which exist in
        more than one CSV are not completely equal, because some 
        characters are different (e.g. long hyphen vs. short hyphen).
        Ideas: Edit distance
        
    """
    task_material_sr  = _load_task_material_sr(csv_path_sr)
    task_material_nr  = _load_task_material_nr(csv_path_nr)
    task_material_tsr = _load_task_material_tsr(csv_path_tsr)
    duplicate_sentences = _load_task_material_duplicate_sentences(csv_path_ds)
    
    # labels can be dropped for the texts table
    # => extract sentences and their ids from the 
    # material loaded above
    sr_sentences = (
        task_material_sr
        [["sentence_id","sentence","sentiment_label","is_control"]]
        .pipe(upd.implode, implode_cols="sentiment_label",
              return_length_1_lists=True) 
        .rename(columns={"sentiment_label": "_labels"})
        # can't group by NaNs, thus assign paragraph_id=nan AFTER implode
        .assign(paragraph_id = np.nan,
                task = "SR")
    )
    nr_sentences = (
        task_material_nr
        [["sentence_id","paragraph_id","sentence","relation_types","is_control"]]
        .pipe(upd.implode, implode_cols="relation_types",
              return_length_1_lists=True)
        .rename(columns={"relation_types": "_labels"})
        .drop_duplicates()
        .assign(task = "NR")
    )
    tsr_sentences = (
        task_material_tsr
        [["sentence_id","paragraph_id","sentence","relation_type","is_control"]]
        .pipe(upd.implode, implode_cols="relation_type",
              return_length_1_lists=True)
        .rename(columns={"relation_type": "_labels"})
        .drop_duplicates()
        .assign(task = "TSR")
    )
    
    # There are some hickups in one of the CSV files, here: duplicates file
    # We will later have to join not only by sentence id and 
    # paragraph id, but also by the sentence itself, because
    # the sentence_id, paragraph_id combinations are not unique
    # in the duplicates file
    # Furthermore: The texts do not all match exactly between duplicates file 
    # and TSR task materials.
    # => for each text in the duplicates file, find the closest
    # match in the TSR sentences and overwrite the sentence column
    # with that.
    duplicate_sentences["closest_tsr_sentence"] = (
        duplicate_sentences
        .sentence
        .apply(lambda x: difflib.get_close_matches(
            x, tsr_sentences.sentence, cutoff=.9)[0]) # speed up by cutoff
    )
    # Some sentences are used in both NR and TSR
    # => make sure, they get the same text_id even
    # if the texts don't match perfectly (which they do not)
    tsr_sentences = tsr_sentences.merge(
        duplicate_sentences[["sid_nr","pid_nr","sid_tsr",
                             "pid_tsr","closest_tsr_sentence"]],
        left_on=["sentence_id","paragraph_id","sentence"],
        right_on=["sid_tsr","pid_tsr","closest_tsr_sentence"], 
        how="left"
    ).drop(columns=["sid_tsr","pid_tsr","closest_tsr_sentence"])
    
    texts_table = (
        pd.concat([sr_sentences, nr_sentences, tsr_sentences], 
                  ignore_index=True, sort=True)
        .sort_values(["task","sentence_id","paragraph_id"])
        .rename(columns = {
            "sentence_id": "_sentence_id",
            "paragraph_id": "_paragraph_id",
            "is_control": "_is_control",
            "sentence": "text",
            "task": "task_id"
        })
    )
    texts_table.loc[:, "length"] = texts_table.text.apply(
        lambda t: len(t.split())
    )
    
    # generate text ids 
    text_id_table = (
        texts_table
        # do not generate text_ids for texts that 
        # are in TSR and NR twice
        .loc[texts_table.sid_nr.isnull(), ["text"]]
        .drop_duplicates()
        .assign(text_id = lambda df: range(df.shape[0]))
    )
    texts_table = texts_table.merge(
        text_id_table, on="text", how="left"
    )
    
    text_ids_from_nr = (
        texts_table
        .loc[~texts_table.text_id.isnull()]
        .loc[texts_table.task_id == "NR"]
        [["text_id","_sentence_id","_paragraph_id"]]
        .rename(columns={
            "text_id":"text_id_from_nr",
            "_sentence_id": "sid_nr",
            "_paragraph_id": "pid_nr",
            "is_control": "_is_control"
        })
    )
    
    # Cannot join with the new pd.NA in a column.
    texts_table.loc[texts_table["sid_nr"].isnull(), "sid_nr"] = np.nan
    texts_table.loc[texts_table["pid_nr"].isnull(), "pid_nr"] = np.nan
    
    # insert text_ids from NR rows to TSR rows where sid_nr is not null
    texts_table = texts_table.merge(
        text_ids_from_nr,
        on=["sid_nr","pid_nr"],
        how="left"
    )
    texts_table.loc[texts_table["text_id"].isnull(), "text_id"] = (
        texts_table.loc[texts_table["text_id"].isnull(), "text_id_from_nr"]
    )
    
    # drop unwanted stuff and change some types
    texts_table = (
        texts_table
        [["text_id","text","task_id","_sentence_id","_paragraph_id","_labels",
          "_is_control","length"]]
        .astype({
            "text_id": int,
            "_paragraph_id": "Int64"
        })
    )
        
    return texts_table

def words_table_from_many_mats(mat_paths: List[str], 
                               text_id_lookup: Union["TextIDLookup",None] = None,
                               subj_task_id_from_mat_path = "default",
                               verbose: bool=True):
    """
    Generates the "words"-table from ZuCo data.
    
    Args:
        mat_paths: (list of str)
            list of paths pointing to .mat-files from ZuCo. 
        verbose: 
            Whether or not to print progress
        subj_task_id_from_mat_path:
            Either...
                "default", which assumes the original filenames
                like "resultsABC_DEF" for which it would infer
                the subject_id 'ABC' and task_id 'DEF'
            or...                    
                Some function with signature f(mp:str) -> (str,str) 
                where the input (mp) is a .mat-filepath and the
                output consists of subject_id and task_id that
                are to be used as values for the respective 
                columns in the final fixations_table.
        text_id_lookup (TextIDLookup or None):
            instance of TextIDLookup
        
    Returns:
        pandas.DataFrame with columns:
            word, word_pos, area_width, area_height, area_left_x, 
            area_bottom_y, area_right_x, area_top_y
    """
    if subj_task_id_from_mat_path == "default":
        subj_task_id_from_mat_path = _subject_and_task_id_from_filename
        
    words_tables = [] # one pandas DF per mat-file, thus one per subject and task
    
    for i_path, mp in enumerate(tqdm(mat_paths, disable=not verbose,
                                     desc="making words table")):
        subject_id, task_id = subj_task_id_from_mat_path(mp)
        granules = [] # will store many pandas DFs, one per text
        
        with h5py.File(mp, mode="r") as mf:
            n_texts = _number_of_texts(mf)
            
            for i_text in range(n_texts):
                wtab_granule = _words_table_granule_from_mat(mf, i_text)
                
                if wtab_granule.shape[0] > 0:
                    # determine (global) text_id
                    if text_id_lookup is None:
                        text_id = None
                    else:
                        text = _get_from_sentenceData(
                            mf, "content", i_text, is_str=True)
                        text_id, is_exact_match = text_id_lookup.lookup(text)
                    wtab_granule.loc[:, "text_id"] = text_id
                else:
                    wtab_granule.loc[:, "text_id"] = []
                
                granules.append(wtab_granule)
        
        wtab = pd.concat(granules, ignore_index=True)
        words_tables.append( wtab )
        
    words_table = pd.concat(words_tables, ignore_index=True)
    words_table = words_table.drop_duplicates()
    
    return words_table


class TextIDLookup:
    """
    Identifies the text_id for a given text.
    
    However, there might arise encoding errors while texts are loaded from the 
    mat-files (e.g.). In order to get a text-id after all, this class supports 
    "loose" lookups. If an exact match can't be found, a close match can be 
    looked for instead (see arg `allow_unexact`).
    
    Args:
        texts_table: (pandas.DataFrame)
            texts-table as returned by (e.g.) function 
            `texts_table_from_task_material`
        allow_unexact: (bool)
            if no exact match can be found (bitwise equal), the text_id of the 
            closest matching text is returned. Lookups for non-matching texts 
            will return faster, as no unexact matching is performed.
        min_sim_ratio: (float)
            minimum similarity ratio (0..1) for unexact matches. If the closest 
            match is below, None will be returned. Ignored if allow_unexact is 
            False. Defaults to 0.95, which yields good results.
            
    Details:
        * closest match is found by `difflib.get_close_matches`
        * min_sim_ratio is passed to `difflib.get_close_matches` as `cutoff`
    """
    def __init__(self, texts_table: pd.DataFrame,
                 allow_unexact: bool=True,
                 min_sim_ratio: float=.95):
        
        self.lookup_table = (
            texts_table
            .copy() # avoid side effects
            .loc[:,["text_id","text"]]
            .drop_duplicates()
            .set_index("text")
        )
        self.allow_unexact = allow_unexact
        self.min_sim_ratio = min_sim_ratio
        
        # make sure lookups will be 1:1 (any text points to only one text_id)
        if not self.lookup_table.index.is_unique:
            raise ValueError("text column is not unique")
            
    def lookup(self, text:str) -> Tuple[Union[int,None], bool]:
        """
        Args:
            text: (str) The text for which the text id is to be determined.
            
        Returns: 
            Tuple[int or None, bool] `(text_id, is_exact_match)`
            text-id will be None if no match (neither exact nor unexact)
            was found.
        """
        tid = self.lookup_exact(text)
        
        if tid is not None:
            return tid, True
        elif self.allow_unexact:
            close_matches = difflib.get_close_matches(
                text, self.lookup_table.index.to_list(), 
                cutoff=self.min_sim_ratio
            )
            
            if len(close_matches) > 0:
                closest_matching_text = close_matches[0]
                tid = self.lookup_exact(closest_matching_text)
                return tid, False
        else:
            return None, False
    
    def lookup_exact(self, text:str) -> Tuple[Union[int,None]]:
        try:
            tid = (self.lookup_table
                   .loc[text]
                   .text_id )
            return tid
        except KeyError: # no exact match found
            return None


# ----------------------------------------
def _words_table_granule_from_mat(mf: h5py.File, i_text: int) -> pd.DataFrame:
    """
    Generates the words-table for a single text in a single ZuCo-mat-file.
    
    Args:
        mf: h5py.File of .mat-file to process
        i_text: number of text that is to be processed
        
    Returns:
        pandas.DataFrame with columns:
            word, word_pos, area_width, area_height, area_left_x, 
            area_bottom_y, area_right_x, area_top_y
    """
    # words might be None, if data not available (seen in ZuCo2)
    words   = _get_from_word_data(mf, "content", i_text, is_str=True)
    no_data = words is None
    if not no_data:
        bounds = _get_from_sentenceData(mf, "wordbounds", i_text).T

        x_left, y_top, x_right, y_bottom = (bounds[:,0], bounds[:,1], 
                                            bounds[:,2], bounds[:,3])
    
    words_granule = pd.DataFrame({
        "word"          : [] if no_data else words,
        "word_pos"      : [] if no_data else list(range(len(words))),
        "area_width"    : [] if no_data else (x_right - x_left),
        "area_height"   : [] if no_data else (y_bottom - y_top),
        "area_left_x"   : [] if no_data else x_left,
        "area_bottom_y" : [] if no_data else y_bottom,
        "area_right_x"  : [] if no_data else x_right,
        "area_top_y"    : [] if no_data else y_top
    })
    
    return words_granule

def _fixations_table_granule_from_mat(mf: h5py.File, i_text: int, 
                                      min_fix_duration:float=100):
    """
    Computes the fixation_table for a single text in a .mat-file. 
    
    Args:
        min_fix_duration: 
            Minimum fixation duration in ms. Fixations with shorter
            durations are filtered out.
    
    Returns:
        pandas.DataFrame with columns:
            duration: (float) duration of fixation in ms
            fix_pos: (int) Determines the order of fixations,
                     start=0, step=1.
            x:
            y: (upside down?)
            
    Details:
        The durations given in the .mat-files are measured 
        in samples. One sample is 2 ms long [1].
        
        [1] https://osf.io/q3zws/wiki/home/
    """
    wordbounds = _get_from_sentenceData(mf, "wordbounds", i_text)
    
    # xs might be None, if data not available (seen in ZuCo2)
    xs      = _get_from_allFixations(mf, "x", i_text)
    no_data = xs is None
    if not no_data:
        ys         = _get_from_allFixations(mf, "y", i_text)
        durations  = _get_from_allFixations(mf, "duration", i_text) * 2
    
        # filter out too short fixations
        xs        = xs[durations >= min_fix_duration]
        ys        = ys[durations >= min_fix_duration]
        durations = durations[durations >= min_fix_duration]

        # assign fixations to words
        wordbounds    = [tuple(row) for row in wordbounds.T]
        fix_points    = list(zip(xs, ys))
        fixated_words = uetc.assign_points_to_rect_areas(fix_points, wordbounds)
    
    fixations_table = pd.DataFrame({
        "fix_pos":  [] if no_data else range(len(xs)),
        "x":        [] if no_data else xs,
        "y":        [] if no_data else ys,
        "word_pos": [] if no_data else fixated_words,
        "duration": [] if no_data else durations
    })
    
    # In rare cases, a single fixation is assigned to two 
    # words (fix_point exactly on edge between two words).
    # => "Explode" these rows. 
    fixations_table = fixations_table.explode("word_pos")
    
    # convert word_pos to nullable integer type
    fixations_table = fixations_table.astype({"word_pos": "Int64"})
    
    return fixations_table

def _subject_and_task_id_from_filename(mat_path: str) -> Tuple[str,str]:
    """
    Returns:
        Tuple[str,str] subject_id, task_id
    """
    filename:str        = os.path.basename(mat_path)
    is_answers_mat:bool = uetc.like(
        filename, r"^Fullresults_(S|N|TS)R_[A-Z]{3}(_T(1|2))?.mat$")
    is_main_results_mat:bool = uetc.like(
        filename, r"^results[A-Z]{3}_(S|N|TS)R.mat$")
        
    if is_answers_mat:
        # original answer-mat filenames are e.g. 'Fullresults_SR_ZKB_T2.mat'
        task_id_pattern    = r"(?<=Fullresults_)[A-Z]+"
        subject_id_pattern = r"(?<=R_)[A-Z]+" # all task_ids end on "R"
    elif is_main_results_mat:
        # original mat filenames are e.g. 'resultsZPH_TSR.mat'
        task_id_pattern    = r"(?<=_)[A-Z]+"
        subject_id_pattern = r"(?<=results)[A-Z]+"
    else:
        raise ValueError("unknown filename pattern")
        
    task_id    = uetc.extract(filename, task_id_pattern)
    subject_id = uetc.extract(filename, subject_id_pattern)
    return subject_id, task_id


# ----------------------------------------
def _get_from_sentenceData(mat_file: h5py.File, dataset_name: str,
                           i_text: int, is_str: bool=False) -> np.ndarray:
    
    ref = mat_file.get("sentenceData").get(dataset_name)[ i_text ][0]
    ds  = mat_file.get(ref)
    if is_str:
        return _h5py_dataset_to_str(ds)
    else:
        return np.array(ds)

def _get_from_word_data(mat_file: h5py.File, dataset_name: str, i_text: int, 
                        is_str:bool=False) -> Union[np.ndarray, None]:
    
    grp_ref = mat_file.get("sentenceData/word")[ i_text ][0]
    grp     = mat_file.get(grp_ref)
    if "MATLAB_fields" in grp.attrs.keys():
        ds = grp.get(dataset_name)
        if is_str:
            ds = [_h5py_dataset_to_str(mat_file.get(elem[0])) for elem in ds]
        else:
            ds = [np.array(mat_file.get(elem[0])) for elem in ds]
        return ds
    else:
        return None

def _get_from_allFixations(mat_file: h5py.File, dataset_name: str, i_text: int
                          ) -> Union[np.ndarray, None]:
    
    ref_allFix_i = mat_file.get("sentenceData/allFixations")[ i_text ][0]
    allFix_i = mat_file.get(ref_allFix_i)
    if "MATLAB_fields" in allFix_i.attrs.keys():
        dataset = allFix_i.get(dataset_name)
        dataset = np.array(dataset).flatten()
        return dataset
    else:
        return None

def _number_of_texts(mat_file: h5py.File):
    return mat_file["sentenceData/content"].size

def _h5py_dataset_to_str(ds):
    if ds.size > 1:
        return "".join(chr(c[0]) for c in ds)
    else:
        return chr(ds[0][0])
    
# ---------------------
# task materials
def _load_task_material_sr(csv_path: str) -> pd.DataFrame:
    """
    For Zuco 1 only.
    """
    task_material_sr = pd.read_csv(csv_path, delimiter=";", 
                                   header=0, comment="#")
    # some lines in the original csv file don't have
    # all 4 lines. correct for that
    rows_with_missing_label = task_material_sr.sentiment_label.isna()
    task_material_sr.loc[rows_with_missing_label,"sentiment_label"] = (
        task_material_sr["control"][rows_with_missing_label]
    )
    task_material_sr.loc[rows_with_missing_label,"control"] = np.nan
    
    # for convenience: "translate" control column to bool
    task_material_sr.loc[:, "is_control"] = (
        task_material_sr["control"]
        .apply(lambda x: True if x=="CONTROL" else False)
    )

    # labels are sometimes float, sometimes int, sometimes str (wtf)
    task_material_sr = task_material_sr.astype({"sentiment_label": int})

    return task_material_sr

def _load_task_material_nr(csv_path: str) -> pd.DataFrame:
    """
    For Zuco 1 only.
    """
    task_material_nr = pd.read_csv(
        csv_path, delimiter=",", engine="python",
        comment="#")
    
    # for convenience: "translate" control column to bool
    task_material_nr.loc[:, "is_control"] = (
        task_material_nr.control
        .apply(lambda x: True if x=="CONTROL" else False)
    )
    return task_material_nr

def _load_task_material_tsr(csv_path: str) -> pd.DataFrame:
    """
    For Zuco 1 only.

    Warning:
        (sentence_id, paragraph_id) is not unique!
        (sentence_id, paragraph_id, relation_type) is.
    """
    first_line = uio.read_lines(csv_path)[0]
    if first_line != "sentence_id;sentence;relation-type":
        warnings.warn("first line does not look like TSR file")
    task_material_tsr = pd.read_csv(
        csv_path, delimiter=";", header=None, skiprows=[0],
        names=["sentence_id","paragraph_id","sentence","relation_type"], 
        comment="#")
    
    # for consistency with SR and NR tables
    task_material_tsr.loc[:,"is_control"] = \
        task_material_tsr["relation_type"] == "CONTROL"
    return task_material_tsr

def _load_task_material_duplicate_sentences(csv_path: str) -> pd.DataFrame:
    """
    For Zuco 1 only.
    """
    task_material_duplicate_sentences = (
        pd.read_csv(csv_path)
        .assign(sidnr_tuple = lambda df: 
                    df["sentence-id-normal-reading"]
                    .apply(lambda x: x.strip().split("-")),
                sid_nr = lambda df: df.sidnr_tuple.apply(lambda x: int(x[0])),
                pid_nr = lambda df: df.sidnr_tuple.apply(lambda x: int(x[1])),
                sidtsr_tuple = lambda df:
                    df["sentence-id-task-specific-reading"]
                    .apply(lambda x: x.strip().split("-")),
                sid_tsr = lambda df: df.sidtsr_tuple.apply(lambda x: int(x[0])),
                pid_tsr = lambda df: df.sidtsr_tuple.apply(lambda x: int(x[1])),
               )
        [["sentence","sid_nr","pid_nr","sid_tsr","pid_tsr"]]
        .astype({
            "sid_nr":"Int64","pid_nr":"Int64",
            "sid_tsr":"Int64","pid_tsr":"Int64"
        })
    )
    return task_material_duplicate_sentences


class ZucoLoader(object):
    def __init__(self,zuco1_prepr_path=None,
                       zuco2_prepr_path=None):
        
        # Load base df
        self.texts_table, words_table, fixations_table = load_all_tables_from_preprocessed(
                    zuco1_prepr_path=zuco1_prepr_path,
                    zuco2_prepr_path=zuco1_prepr_path)

        self.df = make_word_trt_tuple_table(fixations_table, words_table)

    def get_zuco_task_df(self, zuco_version, task):
        
        assert task in ['TSR', 'SR', 'NR']
        
        if zuco_version == 1:
            subjects = all_subjects 
        else:
            #Zuco 2 not fully tested yet
            subjects = all_subjects2
        
        df_task = self.df.query('_zuco_version=={} and subject_id in @subjects and task_id=="{}"'.format(zuco_version,task))
        df_task = df_task.reset_index()
        df_task = df_task.rename(columns={'label':'labels', 'word':'words'})
        return df_task
