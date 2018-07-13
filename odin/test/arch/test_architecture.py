from odin.arch import architecture

import pytest
import torch

def test_init():

    l = architecture.Architecture(0.001)

    assert l.learning_rate == 0.001
    assert l.model == None

def test_compile():

    l = architecture.Architecture(0.001)

    with pytest.raises(NotImplementedError):
        l.compile(None)

def test_pass():

    l = architecture.Architecture(0.001)

    with pytest.raises(TypeError):
        l._pass(None, None)

def test_predict():

    l = architecture.Architecture(0.001)

    with pytest.raises(TypeError):
        l.predict(None)
