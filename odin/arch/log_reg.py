from odin.arch.architecture import Architecture

import torch
import torch.nn as nn

class LogReg(Architecture):

	def __init__(self, learning_rate = 0.001):

		Architecture.__init__(self, learning_rate)

	def compile(self, input_dim):

		self.model = nn.Linear(input_dim, 2)
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate)
		self.criterion = nn.CrossEntropyLoss()
