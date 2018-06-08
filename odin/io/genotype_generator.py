import numpy as np
from keras.utils import Sequence

class GenotypeGenerator(Sequence):

	def __init__(self, x, y, chrs):

		self.x = x
		self.y = y
		blocks = np.insert(np.where(np.diff(chrs) != 0)[0], 0, 0)
		blocks = np.append(blocks, self.x.shape[1])
		blocks = [ (blocks[i],blocks[i+1]) for i in range(len(blocks) - 1) ]
		self.__blocks = [ np.repeat(1,x[1] - x[0]) for x in blocks ]
		self.__N = self.y.shape[0]
		self.__batch_size = int(self.__N/10)

	def __len__(self):
		return int(self.__N/self.__batch_size)

	def __getitem__(self, idx):

		select = np.random.binomial(1, 0.5, len(self.__blocks))
		mask1 = [ bool(i*b) for i,B in zip(select,self.__blocks) for b in B ]
		mask2 = [ not b for b in mask1 ]

		X = []
		Y = []

		for y in [[1,0],[0,1]]:
			pheno = [ bool(x) for x in np.sum(self.y == y, axis = 1) ]

			x1 = self.x[pheno,:]
			x2 = x1
			np.random.shuffle(x2)
			x1[:,mask1] = 0
			x2[:,mask2] = 0

			X.append(x1 + x2)
			Y.append(self.y[pheno])

		return (np.row_stack((X[0], X[1])), np.row_stack((Y[0], Y[1])))
