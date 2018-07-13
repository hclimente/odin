import gzip
import numpy as np
import os
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset

class GWAS(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, file_name, categorical = True):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.x, self.y, self.snps = self._read_file(file_name)

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):

		return (self.x[idx,:], self.y[idx])

	def _read_file(self, file_name):

		name,ext = os.path.splitext(file_name)
		txt = {'.tsv': '\t', '.csv': ','}

		if ext in txt.keys():
			x, y, snps = self._read_table(file_name, txt[ext])
		elif ext == '.ped' or ext == '.map':
			x, y, snps = self._read_ped(name)
		else:
			raise IOError

		x = torch.from_numpy(x)
		x = x.type(torch.FloatTensor)
		y = torch.from_numpy(y)

		return (x, y, snps)

	def _read_table(self, file_name, sep):
		gwas = pd.read_csv(file_name, sep = sep)

		y = gwas.pop('y').values
		x = gwas.values
		snps = gwas.columns

		return x,y,snps

	def _read_ped(self, name):

		encoding = { 'AA': 2, 'AT': 1, 'AC': 1, 'AG': 1,
					 'TA': 1, 'TT': 3, 'TC': 1, 'TG': 1,
					 'CA': 1, 'CT': 1, 'CC': 4, 'CG': 1,
					 'GA': 1, 'GT': 1, 'GC': 1, 'GG': 5 }
		x = []
		y = []

		with open(name + '.ped', 'r') as PED:
			for line in PED:
				line = line.strip().split(' ')
				chr1 = line[6:][0::2]
				chr2 = line[6:][1::2]

				x.append([ encoding[a1+a2] for a1,a2 in zip(chr1, chr2) ])
				y.append(int(line[5]))

		x = np.array(x, dtype = 'uint8')
		x = np.apply_along_axis(recode, 0, x)
		y = np.array(y)

		snps = pd.read_csv(name + '.map', sep = '\t', header = None)
		snps.columns = ['chr','snp','cm','pos']
		snps = snps.pop('snp')

		return x,y,snps

def recode(c):
	gt = np.unique(c)

	if len(gt) == 2:
		c[c == gt[1]] = 0
	else:
		i = np.count_nonzero(c == gt[2]) > np.count_nonzero(c == gt[1])
		c[c == gt[1 + i]] = 0
		c[c == gt[1 + (not i)]] = 2

	return c
