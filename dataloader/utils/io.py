import os
from typing import List

def write_list(l: List[str], fpath: str, enc_ascii: bool=False,
               ascii_errors: str="replace") -> None:
    """
    Writes strings from a list as lines to a file.

    Args:
        l: list to write to fpath
        fpath: path of file to write to
    """
    dirpath = os.path.dirname(fpath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    if enc_ascii:
        with open(fpath, "wb") as f:
            for elem in l:
                f.write((elem + "\n").encode(encoding="ascii", 
                                             errors=ascii_errors))
    else:
        with open(fpath, "w") as f:
            for elem in l:
                f.write("%s\n" % elem)
            
def read_lines(fpath: str) -> List[str]:
    """
    Reads lines from a file as text and returns these as a list.
    """
    with open(fpath, "rt") as f:
        file_content = f.readlines()
    # remove "\n" at the end of each line
    file_content = [line.rstrip("\n") for line in file_content]
    return file_content
