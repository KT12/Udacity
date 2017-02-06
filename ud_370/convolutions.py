from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels  = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels  = save['valid_labels']
	test_dataset  = save['test_dataset']
	test_labels   = save['test_labels']
	del save
	print('Training set', train_dataset.shape, train_labels)
	print('Validation set', valid_dataset.shape, valid_labels)
	print('Test set', test_dataset.shape, test_labels)

image_size = 28
num_labels = 10
num_channels = 1 # combine RGB into grayscale

def reformat(dataset, labels):
	datset = dataset.reshape((-1, image_size, image_size, 
		num_channels)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
			/ predictions.shape[0])

# Build network with 2 convolutional layers

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()
	with graph.as_default():

		tf.set_random_seed(1)

		# Input data
		tf_train_dataset = tf.placeholder(tf.float32, shape=
			(batch_size, image_size, image_size, num_channels))
		tf_train_labels  = tf.placeholder(tf.float32, shape=
			(batch_size, num_labels))
		