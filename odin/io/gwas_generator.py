from odin.io.gwas import GWAS

import numpy as np
import os
import torch
import pandas as pd

class GWASGenerator(GWAS):
	"""Face Landmarks dataset."""

	def __init__(self, file_name, chrs):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.x, self.y, self.snps = self._read_file(file_name)
		blocks = np.insert(np.where(np.diff(chrs) != 0)[0], 0, 0)
		blocks = np.append(blocks, self.x.shape[1])
		blocks = [ (blocks[i],blocks[i+1]) for i in range(len(blocks) - 1) ]
		self.__blocks = [ np.repeat(1,x[1] - x[0]) for x in blocks ]

	def __len__(self):

		N = 0
		for y in np.unique(self.y):
			N += np.power(np.sum(self.y.numpy() == y), len(self.__blocks))

		return N

	def __getitem__(self, idx):

		select = np.random.binomial(1, 0.5, len(self.__blocks))
		mask1 = [ bool(i*b) for i,B in zip(select,self.__blocks) for b in B ]
		mask2 = [ not b for b in mask1 ]

		itertools.product(a, b)

		idx1, idx2 = oned2two2(idx)

		x1 = self.x[idx1, mask1]
		x1[mask2] = 0
		x2 = self.x[idx2, mask2]
		x2[mask1] = 0

		x = x1 + x2
		y = np.unique(y[idx1], y[idx2])

		return x, y
