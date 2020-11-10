import numpy as np

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def derivativeSigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))


def MSEloss(y_actual, y_pred):
	"""
	Mean squared error: squared of actual target minus predicted target
	y_actual = expected output
	y_pred = predicted output
	"""
	return (y_actual - y_pred) ** 2



class twoLNetwork:
	def __init__(self, inputs, hidden, output):
		"""
		w1 = weights connecting input and hidden layer
		w2 = weights connecting hidden and output layer
		b1 = bias in hidden layer activation function
		b2 = bias in output layer activation funtion
		"""
		self.inputs = inputs
		self.hidden = hidden
		self.output = output
		self.w1 = np.random.rand(self.hidden, self.inputs) - 0.5
		self.w2 = np.random.rand(self.output, self.hidden) - 0.5
		self.b1 = np.random.rand(self.hidden, 1) - 0.5
		self.b2 = np.random.rand(self.output, 1) - 0.5
		pass
	def predict(self, X, y):
		"""
		X = input training data
		"""
		z1 = np.dot(self.w1, X) + self.b1
		a1 = sigmoid(z1)

		z2 = np.dot(self.w2, a1)
		a2 = sigmoid(z2)

		return a2

	def train(self, X, y, alpha):
		"""
		X = input data
		y = expected output
		alpha = learning rate
		"""
		z1 = np.dot(self.w1, X) + self.b1
		a1 = sigmoid(z1)

		z2 = np.dot(self.w2, a1)
		a2 = sigmoid(z2)

		err = MSEloss(y, a2)
		#parameters update
		da2 = 2 * (a2 - y)

		db2 = da2 * derivativeSigmoid(z2)
		dw2 = np.dot(db2, a1.T)
		db1 = np.dot(self.w2.T, da2 * derivativeSigmoid(z2)) * derivativeSigmoid(z1)
		dw1 = np.dot(db1, X.T)

		self.w1 = self.w1 - alpha * dw1
		self.b1 = self.b1 - alpha * db1
		self.w2 = self.w2 - alpha * dw2
		self.b2 = self.b2 - alpha * db2

		return err

import sys, numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[0:1000].reshape(1000,28*28) / 255, y_train[0:1000])
one_hot_labels = np.zeros((len(labels),10))
for i,l in enumerate(labels):
	one_hot_labels[i][l] = 1
labels = one_hot_labels
test_images = x_test.reshape(len(x_test),28*28) / 255
test_labels = np.zeros((len(y_test),10))

for i,l in enumerate(y_test):
	test_labels[i][l] = 1

a = twoLNetwork(784, 100, 10)
p= np.expand_dims(test_labels[0], axis = 1)

for i in range(1001):
	error = 0
	for j in range(len(images)):
		y_actual = np.expand_dims(labels[j], axis = 1)
		data = np.expand_dims(images[j], axis = 1)
		error += a.train(data, y_actual, 0.01)
	print('error :', error)

#test model

error = 0
for i in range(len(test_images)):
	y_actual = np.expand_dims(test_labels[j], axis = 1)
	data = np.expand_dims(test_images[j], axis = 1)
	y_pred = a.predict(data, y_actual)
	error += MSEloss(y_actual, y_pred)

print('error :' ,error)