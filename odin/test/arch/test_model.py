import odin
from odin.arch import model
from odin.io import io

import gzip
import os
import pickle
import pytest
import torch

scriptPath = os.path.realpath(__file__)
dataPath = os.path.dirname(scriptPath) + "/../data/"

def test_init():

    m = model.Model('logreg')

    assert type(m._arch) == odin.arch.log_reg.LogReg
    assert m._arch.model == None

def test_fit():

    m = model.Model('logreg')
    g = odin.io.gwas.GWAS(dataPath + 'gt.tsv')

    m.fit(g, 1, 3)

    assert type(m._arch.model) == torch.nn.modules.linear.Linear

def test_predict():
    pass

def test_save():

    m = model.Model('logreg')
    g = odin.io.gwas.GWAS(dataPath + 'gt.tsv')
    m.fit(g, 1, 3)

    io.save_pickle(m, 'model.pkl')
    reloaded = io.load_pickle('model.pkl')

    assert type(reloaded._arch.model) == torch.nn.modules.linear.Linear

    os.remove('model.pkl')
