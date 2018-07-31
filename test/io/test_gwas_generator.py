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

    assert (csv.x == tsv.x).all()
    assert (csv.y == tsv.y).all()
    assert (csv.snps == tsv.snps).all()
    assert tsv.__len__() == 32

def test_read_file():

    csv = GWASGenerator(dataPath + 'gt.csv', [1,1,1,2,2,2])

    # csv
    x_csv, y_csv, snps_csv = csv._read_file(dataPath + 'gt.csv')

    # assert y_csv
    assert (y_csv == 1).sum() == 4
    assert (y_csv == 0).sum() == 4

    # assert x_csv
    assert x_csv.shape == (8, 5)
    assert (x_csv == 0).sum() == 15
    assert (x_csv == 1).sum() == 15
    assert (x_csv == 2).sum() == 10

    # assert snps_csv
    assert (snps_csv == np.array(['rs1','rs2','rs3','rs4','rs5'])).all()

    # tsv
    x_tsv, y_tsv, snps_tsv = csv._read_file(dataPath + 'gt.tsv')

    assert (x_tsv == x_csv).all()
    assert (y_tsv == y_csv).all()
    assert (snps_tsv == snps_csv).all()

    with pytest.raises(IOError):
        csv._read_file(dataPath + 'gt.kkk')

def test__len__():

    csv = GWASGenerator(dataPath + 'gt.csv', [1,1,1,2,2,2])

    assert csv.__len__()

def test__getitem__():

    csv = GWASGenerator(dataPath + 'gt.csv', [1,1,1,2,2,2])

    control = torch.tensor([0])
    case = torch.tensor([1])

    assert csv.__getitem__(0)[1] == control
    assert csv.__getitem__(16)[1] == control
    assert csv.__getitem__(17)[1] == case
    assert csv.__getitem__(32)[1] == case
    
    with pytest.raises(KeyError):
        csv.__getitem__(33)
