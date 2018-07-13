from odin.io.gwas_generator import GWASGenerator

import numpy as np
import os
import pytest
import torch
from torch.utils.data import Dataset

scriptPath = os.path.realpath(__file__)
dataPath = os.path.dirname(scriptPath) + "/../data/"

def test_init():

    csv = GWASGenerator(dataPath + 'gt.csv', [1,1,1,2,2,2])
    tsv = GWASGenerator(dataPath + 'gt.tsv', [1,1,1,2,2,2])

    assert issubclass(GWASGenerator, Dataset)

    assert np.all(csv.x == tsv.x)
    assert np.all(csv.y == tsv.y)
    assert np.all(csv.snps == tsv.snps)
    assert tsv.__len__() == 32

def test_read_file():

    csv = GWASGenerator(dataPath + 'gt.csv', [1,1,1,2,2,2])

    # csv
    x_csv, y_csv, snps_csv = csv._read_file(dataPath + 'gt.csv')

    # assert y_csv
    assert torch.sum(y_csv == 1) == 4
    assert torch.sum(y_csv == 0) == 4

    # assert x_csv
    assert x_csv.shape == (8, 5)
    assert torch.sum(x_csv == 0) == 15
    assert torch.sum(x_csv == 1) == 15
    assert torch.sum(x_csv == 2) == 10

    # assert snps_csv
    assert np.all(snps_csv == np.array(['rs1','rs2','rs3','rs4','rs5']))

    # tsv
    x_tsv, y_tsv, snps_tsv = csv._read_file(dataPath + 'gt.tsv')

    assert np.all(x_tsv == x_csv)
    assert np.all(y_tsv == y_csv)
    assert np.all(snps_tsv == snps_csv)

    with pytest.raises(IOError):
        csv._read_file(dataPath + 'gt.kkk')

def test__getitem__():

    pass
