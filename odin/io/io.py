from keras.utils import to_categorical
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

import os
import pickle

def save_pickle(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def load_pickle(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1

    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)

    return obj

def read_x_y(filepath, categorical = False):

	gwas = np.load(filepath)

	x = gwas[:,1:]
	y = gwas[:,0] - 1

	if categorical:
		y = to_categorical(y)

	x, y = shuffle(x,y)

	return(x,y)

def read_map(filepath):

	snps = pd.read_csv(filepath, sep = '\t', header = None)
	snps.columns = ['chr','snp','cm','pos']

	return(snps)
