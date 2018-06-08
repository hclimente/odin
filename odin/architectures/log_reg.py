from keras.models import Sequential
from keras.layers import Dense

class LogReg():

	def __init__(self):
		pass

	def compile(self, input_dim):

		model = Sequential()
		model.add(Dense(2, activation = 'softmax', input_dim = input_dim))
		model.compile(optimizer = 'sgd',
						   loss = 'binary_crossentropy',
						   metrics = ['binary_accuracy'])
		model.summary()

		return model
