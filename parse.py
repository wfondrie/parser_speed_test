"""
Speed test parsing methods for Alan
"""
import random
import string
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd


# Functions -------------------------------------------------------------------
def random_data(out_root, n_rows=100, n_cols=30000):
    """Create a random dataset"""
    out_file = f"{out_root}_{n_rows}-{n_cols}.txt"
    if Path(out_file).exists():
        return out_file

    # Create directory if necessary:
    Path(out_file).parent.mkdir(exist_ok=True)

    print("Generating int columns...")
    int_cols = np.random.randint(0, 100, size=(n_rows, n_cols-1))

    print("Generating a string column...")
    str_col = np.array([
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
        for _ in range(n_rows)
    ])

    print("Creating a DataFrame...")
    dat = pd.DataFrame({"str_col": str_col})
    dat = pd.concat([dat, pd.DataFrame(int_cols)], axis=1)

    print(f"Saving DataFrame to '{out_file}'...")
    dat.to_csv(out_file, sep="\t", index=False)

    return out_file


def read_line(line):
    return line.rstrip().split("\t")


def with_dtype(txt_file):
    """Similar to Alan's current approach"""
    with open(txt_file) as txt_data:
        header = read_line(txt_data.readline())

    dtypes = {k: np.int for k in header}
    dtypes[header[0]] = np.string_
    return pd.read_csv(txt_file, sep="\t", dtype=dtypes, low_memory=False)


def read_txt(txt_file, workers=1):
    """A relatively simple Python parser"""
    with open(txt_file) as txt_data:
        header = read_line(txt_data.readline())

        if workers > 1:
            with ProcessPoolExecutor(workers) as prc:
                df = pd.DataFrame.from_records(
                    prc.map(read_line, txt_data), columns=header
                )
        else:
            df = pd.DataFrame.from_records(
                map(read_line, txt_data), columns=header
            )

    return df
