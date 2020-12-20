import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

"""
"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
inptsz = 28 * 28
hidsz = 100
outsz = 10

#Activation functions
def ReLU(x):
	return (x > 0) * x

def dReLU(x):
	return 1 * (x > 0)

def softmax(x):
	x -= np.max(x)
	return np.exp(x)/np.sum(np.exp(x))

def sigmoid(x):
	return 1 /(1 + np.exp(-x))

def dsigmoid(x):
	return x * (1 -x)
#Initializing learnable parameters
W_1 = 0.2 * np.random.random((hidsz, inptsz)) - 0.1
W_2 = 0.2 * np.random.random((outsz,hidsz)) - 0.1

b_1 = np.random.randn(hidsz, 1) + 0.01
b_2 = np.random.randn(outsz, 1) + 0.01

lr = 0.001 #Learning rate
#One_hot_Encoding train and test labels
one_hot_labels = np.zeros((len(y_train), 10, 1))
for i, l in enumerate(y_train):
	one_hot_labels[i][l] = 1
y_train = one_hot_labels

test_labels = np.zeros((len(y_test),10,1))
for i,l in enumerate(y_test):
	test_labels[i][l] = 1
#flatening data
y_test = test_labels
x = np.mean(x_train)
x_train = x_train.reshape(60000, 28*28, 1) - x
x_test = x_test.reshape(10000, 28*28, 1) - x

lmda = 0.005
plt.figure()
#Updating parameters
for i in range(1000):
	dW_2 = np.zeros(W_2.shape)
	db_2 = np.zeros(b_2.shape)

	dW_1 = np.zeros(W_1.shape)
	db_1 = np.zeros(b_1.shape)

	loss = 0.0
	reg_loss = 0.0
	count = 0
	for j in range(10000):
		#Forward Propagation
		z_1 = np.dot(W_1, x_train[j]) + b_1
		a_1 = sigmoid(z_1)

		z_2 = np.dot(W_2, a_1) + b_2
		a_2 = softmax(z_2)

		#Regularization loss
		reg_loss += (lmda) * (np.sum(np.square(W_1)) + np.sum(np.square(W_2)))
		#Full Loss
		loss += np.sum(-y_train[j] * np.log(a_2)) + reg_loss
		count += int(np.argmax(a_2) == np.argmax(y_train[j]))

		#Updating parameters
		db_2 = a_2 - y_train[j]
		dW_2 = np.dot(db_2, a_1.transpose())
		db_1 = (np.dot(W_2.transpose(), (a_2 - y_train[j])) * dsigmoid(a_1))
		dW_1 = np.dot(db_1,x_train[j].transpose())

		#Stochastic gradient descent
		W_1 = W_1 - lr * dW_1 - lmda * lr * W_1
		W_2 = W_2 - lr * dW_2 - lmda * lr * W_2
		b_1 = b_1 - lr  * db_1
		b_2 = b_2 - lr  * db_2

	sys.stdout.write("Loss: {}, accuracy: {}".format(loss/float(10000 * 10000), count/float(10000)))
	plt.scatter(i, loss/float(10000 * 10000), c = 'b', marker = '.', linewidth = 0.5)

	
	loss = 0.0
	count = 0
	for j in range(1000):
		#Test set error checking
		z_1 = np.dot(W_1, x_test[j]) + b_1
		a_1 = sigmoid(z_1)

		z_2 = np.dot(W_2, a_1) + b_2
		a_2 = softmax(z_2)

		count += int(np.argmax(y_test[j]) == np.argmax(a_2))
		loss += np.sum(-y_test[j] * np.log(a_2))
	plt.scatter(i, loss/float(1000), c = 'r', marker = '.', linewidth = 0.5)
	sys.stdout.write(" test loss: {} test_accuracy: {}".format(loss/float(1000), count/float(1000)))
	sys.stdout.write("\n")
	#plt.show()

plt.xlabel('epoch')
plt.ylabel('error')
plt.show()
