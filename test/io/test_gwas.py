from odin.io.gwas import GWAS
from odin.io import io

import gzip
import numpy as np
import os
import pickle
import pytest
import torch
from torch.utils.data import Dataset

scriptPath = os.path.realpath(__file__)
dataPath = os.path.dirname(scriptPath) + "/../data/"

def test_init():

    csv = GWAS(dataPath + 'gt.csv')
    tsv = GWAS(dataPath + 'gt.tsv')

    assert issubclass(GWAS, Dataset)

    assert (csv.x == tsv.x).all()
    assert (csv.y == tsv.y).all()
    assert (csv.snps == tsv.snps).all()
    assert tsv.__len__() == 8

def test_read_file_table():

    csv = GWAS(dataPath + 'gt.csv')

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
    assert (snps_csv == np.array(['rs1','rs2','rs3','rs4','rs5'])).all()

    # tsv
    x_tsv, y_tsv, snps_tsv = csv._read_file(dataPath + 'gt.tsv')

    assert (x_tsv == x_csv).all()
    assert (y_tsv == y_csv).all()
    assert (snps_tsv == snps_csv).all()

    with pytest.raises(IOError):
        csv._read_file(dataPath + 'gt.kkk')


def test_read_file_ped():

    ped = GWAS(dataPath + 'gt.ped')

    x,y,snps = ped._read_file(dataPath + 'gt.ped')

    x1,y1 = ped.__getitem__(1)
    assert (x1 == torch.tensor([[1.,1.,1.,0.]])).all()
    assert (y1 == torch.tensor([1])).all()

    x3,y3 = ped.__getitem__(3)
    assert (x3 == torch.tensor([[2.,0.,2.,1.]])).all()
    assert (y3 == torch.tensor([2])).all()

    x5,y5 = ped.__getitem__(5)
    assert (x5 == torch.tensor([[0.,1.,0.,0.]])).all()
    assert (y5 == torch.tensor([2])).all()

    x7,y7 = ped.__getitem__(7)
    assert (x7 == torch.tensor([[2.,1.,2.,0.]])).all()
    assert (y7 == torch.tensor([2])).all()

def test__getitem__():

    csv = GWAS(dataPath + 'gt.csv')

    x0,y0 = csv.__getitem__(0)
    assert (x0 == torch.tensor([[0.,0.,0.,1.,1.]])).all()
    assert (y0 == torch.tensor([1])).all()

    x4,y4 = csv.__getitem__(4)
    assert (x4 == torch.tensor([[1.,1.,1.,1.,1.]])).all()
    assert (y4 == torch.tensor([0])).all()

def test_save():

    csv = GWAS(dataPath + 'gt.csv')
    io.save_pickle(csv, 'gwas.pkl')

    assert os.stat('gwas.pkl').st_size > 0

    reloaded = io.load_pickle('gwas.pkl')

    assert (csv.__getitem__(2)[1] == reloaded.__getitem__(2)[1]).all()
    assert (csv.__getitem__(4)[1] == reloaded.__getitem__(4)[1]).all()
    assert (csv.__getitem__(6)[1] == reloaded.__getitem__(6)[1]).all()

    os.remove('gwas.pkl')
