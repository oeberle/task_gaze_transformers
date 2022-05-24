import pandas as pd

from typing import List, Tuple, Union, Optional


def implode(df: pd.DataFrame, 
            groupby_cols: Optional[Union[str, List[str]]]=None, 
            implode_cols: Optional[Union[str, List[str]]]=None,
            return_length_1_lists: bool=True
           ) -> pd.DataFrame:
    """
    Groups the dataframe and wraps up the columns in each group to tuples. 
    The number of columns stays the same, whereas the number of rows is reduced 
    to the number of groups. For an illustration see the example at the end of
    this docstring.
    
    Meant as counterpart to function `pandas.DataFrame.explode`.
    
    Args:
        groupby_cols (str or list of str): 
            columns to group by (as passed to `pandas.DataFrame.groupby`). Must 
            be None, if implode_cols is not None.
        implode_cols (str or list of str): 
            Columns to implode. All others will be used to group by. Must be 
            None, if groupby_cols is not None.
        return_length_1_lists (bool): 
            Whether to keep lists of length 1 (True) or to turn them into the 
            only scalar value they hold (False). 
    Returns:
        pandas.DataFrame
        
    Example:
        ```
        colA  colB
           a     1
           b     2
           a     3
        ```
           
        grouped by colA, becomes...
        ```
        colA    colB
           a   (1,3)
           b    (2,)
        ```
    """
    if groupby_cols is not None and implode_cols is not None:
        raise ValueError("Either groupby_cols or implode_cols must be passed, "
                         "not both.")

    if groupby_cols is not None:
        if type(groupby_cols) == str:
            groupby_cols = [groupby_cols] # for convenience
        implode_cols = [col for col in df.columns if col not in groupby_cols]
    elif implode_cols is not None:
        if type(implode_cols) == str:
            implode_cols = [implode_cols] # for convenience
        groupby_cols = [col for col in df.columns if col not in implode_cols]
    
    grouped         = df.groupby(groupby_cols)
    imploded_series = []
    for col in implode_cols:
        ser = (
            grouped
            .apply(lambda grp: tuple(grp[col]))
            .rename(col)
        )
        if not return_length_1_lists:
            ser = ser.apply(lambda x: x if len(x) > 1 else x[0])
        imploded_series.append(ser)
        
    return pd.concat(imploded_series, axis=1).reset_index()
