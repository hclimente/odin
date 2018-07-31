from odin.io.gwas import GWAS

import numpy as np
import os
import torch
import pandas as pd

class GWASGenerator(GWAS):
	"""GWAS generator."""

	def __init__(self, file_name, chrs):
		"""
		Args:
			file_name (string): Path to the filename with the genotypes.
			chrs (string): Blocks.
		"""
		self.x, self.y, self.snps = self._read_file(file_name)
		blocks = np.insert(np.where(np.diff(chrs) != 0)[0], 0, 0)
		blocks = np.append(blocks, self.x.shape[1])
		blocks = [ (blocks[i],blocks[i+1]) for i in range(len(blocks) - 1) ]
		self._blocks = [ np.repeat(1,x[1] - x[0]) for x in blocks ]

		self._y_levels = {}

		for y in torch.from_numpy(np.unique(self.y)):
			n = int(np.power((self.y == y).sum(), len(self._blocks)))
			self._y_levels[y] = n

		self.__n = np.sum([ n for y,n in self._y_levels.items() ])

	def __len__(self): return self.__n

	def __getitem__(self, idx):

		y = None
		for lvl,n in self._y_levels.items():
			if idx > n:
				idx -= n
			else:
				y = lvl
				break

		idx1 = int(idx / self._y_levels[y])
		idx2 = idx % self._y_levels[y]

		select = np.random.binomial(1, 0.5, len(self._blocks))
		mask1 = [ bool(i*b) for i,B in zip(select,self._blocks) for b in B ]
		mask2 = [ not b for b in mask1 ]

		x1 = self.x[self.y == y,:][idx1, mask1]
		x1[mask2] = 0
		x2 = self.x[self.y == y,:][idx2, mask2]
		x2[mask1] = 0
		
		x = x1 + x2

		return x, y
