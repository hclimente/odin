from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1

from odin.preprocessing import genotype_generator

class LogReg():

	def __init__(self):
		pass

	def run(self, x, y, chrs):

		model = Sequential()
		model.add(Dense(1000, activation = 'softmax', input_dim = x.shape[1]))
		model.add(Dropout(0.5))
		model.add(Dense(100, activation = 'relu'))
		model.add(Dropout(0.5))
		model.add(Dense(2, activation = 'softmax'))

		G = genotype_generator.GenotypeGenerator(x, y, chrs)

		model.compile(optimizer='sgd',
		              loss='binary_crossentropy',
		              metrics=['accuracy'])
		model.summary()

		model.fit_generator(generator = G, epochs = 10, shuffle = True, steps_per_epoch = 20)
