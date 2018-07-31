from odin.arch import log_reg
from odin.io import gwas

import numpy as np
import os
import pytest
import torch

scriptPath = os.path.realpath(__file__)
dataPath = os.path.dirname(scriptPath) + "/../data/"

def test_init():

    l = log_reg.LogReg(0.25)

    # correct initialization
    assert l.learning_rate == 0.25
    assert l.model == None

def test_compile():

    l = log_reg.LogReg()
    l.compile(5)

    assert type(l.model) == torch.nn.modules.linear.Linear
    assert l.model.in_features == 5
    assert l.model.out_features == 2
    assert type(l.optimizer) == torch.optim.SGD
    assert type(l.criterion) == torch.nn.CrossEntropyLoss

def test_pass():

    g = gwas.GWAS(dataPath + 'gt.tsv')

    l = log_reg.LogReg()
    l.compile(5)

    for x, y in g:
        l._pass(x.unsqueeze(0), y.unsqueeze(0))
