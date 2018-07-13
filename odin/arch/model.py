from odin.arch import *

import gzip
import pickle
from torch.utils.data import DataLoader

class Model:

	def __init__(self, arch):

		self._archs = {'logreg': log_reg.LogReg, 'diet': diet.DietNetwork, 'dnp': dnp.DeepNeuralPursuit}

		self._arch = self._archs[arch]()

	def fit(self, dataset, epochs = 10, batch_size = 100):

		loader = DataLoader(dataset, shuffle = True, batch_size = batch_size)

		x_ex,_ = dataset.__getitem__(0)
		self._arch.compile(x_ex.shape[0])

		total_step = len(loader)
		for epoch in range(epochs):
			for x, y in loader:
				self._arch._pass(x, y)

	def predict(self, x):

		return self._arch.predict(x)
