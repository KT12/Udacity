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
		tf_valid_dataset = tf.constant(valid_dataset)
		tf_test_dataset  = tf.constant(test_dataset)

		# Variables
		layer1_weights = tf.Variable(tf.truncated_normal(
			[patch_size, patch_size, num_channels, depth], 
			stddev=0.1))
		layer1_biases = tf.Variable(tf.zeros([depth]))

		layer2_weights = tf.Variable(tf.truncated_normal(
			[patch_size, patch_size, depth, depth], stddev=0.1))
		layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

		layer3_weights = tf.Variable(tf.truncated_normal(
			[image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
		layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

		layer4_weights = tf.Variable(tf.truncated_normal(
			[num_hidden, num_labels], stddev=0.1))
		layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

		#Model
		def model(data):
			conv_1 = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1],
				padding='SAME')
			hidden_1 = tf.nn.relu(conv_1 + layer1_biases)

			conv_2 = tf.nn.conv2d(hidden_1, layer2_weights, 
				[1, 2, 2, 1], padding='SAME')
			hidden_2 = tf.nn.relu(conv_2 + layer2_biases)

			shape = hidden_2.get_shape().as_list()
			rehsape = tf.reshape(hidden_2, [shape[0], shape[1] * shape[2] *
					shape[3]])

			hidden_3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

			return tf.matmul(hidden_3, layer4_weights) + layer4_biases