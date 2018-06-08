from odin.architectures import *

import pickle

class Model:

	def __init__(self, params):

		self.__archs = {'logreg': log_reg.LogReg, 'diet': diet.DietNetwork, 'dnp': dnp.DeepNeuralPursuit}

		self.__epochs = params.epochs if 'epochs' in params else 10
		self.__steps_per_epoch = params.steps_per_epoch if 'steps_per_epoch' in params else 20
		self.__batch_size = params.batch_size if 'batch_size' in params else 100
		self.__architecture = params.architecture if 'architecture' in params else None

	def train(self, x_train, x_test, y_train, y_test, generator = None):

		self.model = self.__archs[self.__architecture]().compile(x_test.shape[1])

		if generator:
			self.model.fit_generator(generator = generator,
									 epochs = self.__epochs,
									 shuffle = True,
									 steps_per_epoch = self.__steps_per_epoch,
									 validation_data = (x_test,y_test))
		else:
			self.model.fit(x_train, y_train,
						   epochs = self.__epochs,
						   shuffle = True,
						   batch_size = self.__batch_size,
						   validation_data = (x_test,y_test))

	def predict(self, x, y):

		self.model.predict(x, y)

	def save(self, outdir):

		filename = "{}/{}.pkl".format(outdir, self.__architecture)

		with open(filename, "wb") as DUMP:
			pickle.dump(self, DUMP, -1)
