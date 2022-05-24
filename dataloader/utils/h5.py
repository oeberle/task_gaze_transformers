import h5py

def h5file_is_readable(h5_filepath: str):
    """
    Returns `True`, if an .h5 file can be opened, otherwise `False`.
    """
    try:
        f = h5py.File(h5_filepath, mode="r")
        f.close()
        return True
    except OSError:
        return False
