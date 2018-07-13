import torch.nn as nn

import abc

class Architecture():

	def __init__(self, learning_rate):

		self.learning_rate = learning_rate
		self.model = None

	@abc.abstractmethod
	def compile(self, input_dim):
		raise NotImplementedError()

	def _pass(self, x, y):

		# forward pass
		y_pred = self.model(x)
		loss = self.criterion(y_pred, y)

		# Backward and optimize
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def predict(self, x):
		return self.model(x)
