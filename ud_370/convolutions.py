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

		# Train computation
		logits = model(tf_train_dataset)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
				labels=tf_train_labels, logits=logits))

		# Optimizer
		optimizer = tf.train.GradienDescentOptimizer(0.05).minimize(loss)


		# Predictions
		train_prediction = tf.nn.softmax(logits)
		valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
		test_prediction  = tf.nn.softmax(model(tf_test_dataset))

# Run the covnolutional neural network

num_steps = 1001

with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	print('Initialized')
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
		batch_labels = train_labels[offset:(offset + batch_size),:]
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run([optimizer, loss, train_prediction], 
				feed_dict=feed_dict)
		if (step % 100 == 0):
			print('Minibatch loss at step %d: %f' % (step, l))
      		print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      		print('Validation accuracy: %.1f%%' % accuracy(
        		valid_prediction.eval(), valid_labels))
	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# Alter model above to use maxpool with stride 2 and kernel size 2
# instead of convolutions

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
	tf.set_random_seed(1)

	# Input data
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size,
			image_size, num_channels))
	tf_train_labels  = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset  = tf.constant(test_dataset)

	# Variables
	layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size,
		num_channels, depth], stddev=0.1))
	layer1_biases  = tf.Variable(tf.zeros([depth]))

	layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size,
		depth, depth], stddev=0.1))
	layer2_biases  = tf.Variable(tf.constant(1.0, shape=[depth]))

	layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 *
		image_size // 4 * depth, num_hidden], stddev=0.1))
	layer3_biases  = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

	layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels],
		stddev=0.1))
	layer4_biases  = tf.Variable(tf.constant(1.0, shape=[num_labels]))


	def model(data):
		conv_1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
		hidden_1 = tf.nn.relu(conv_1 + layer1_biases)
		max_pool_1 = tf.nn.max_pool(hidden_1, ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1], padding='SAME')

		conv_2 = tf.nn.conv2d(max_pool_1, layer2_weights, [1, 1, 1, 1],
		padding='SAME')
		hidden_2 = tf.nn.relu(conv_2 + layer2_biases)
		max_pool_2 = tf.nn.max_pool(hidden_2, ksize=[1, 2, 2, 1], 
			strides=[1, 2, 2, 1], padding='SAME')

		shape max_pool_2.get_shape().as_list()
		reshape = tf.reshape(max_pool_2, [shape[0], shape[1] * shape[2] *
			shape[3]])
		hidden_3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
		return tf.matmul(hidden_3, layer4_weights) + layer4_biases

		# Training computation
		logits = model(tf_train_dataset)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=tf_train_labels, logits=logits))

		# Optimizer
		optimizer = tf.trian.GradienDescentOptimizer(0.05).minimize(loss)

		# Predictions for training, validation, and test data
		train_prediction = tf.nn.softmax(logits)
		valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
		test_prediction  = tf.nn.softmax(model(tf_test_dataset))

		num_steps = 1001

		with tf.Session(graph=graph) as session:
			tf.initialize_all_variables().run()
			print('Initializaed')
			for step in range(num_steps):
				offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
				batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
				batch_labels = train_labels[offset:(offset + batch_size), :]
				feed_dict = {tf_train_dataset : batch_data,
					tf_train_labels : batch_labels}
				_, l, predictions = session.run([optimizer, loss, train_prediction],
						feed_dict=feed_dict)
				if (step%100 == 0):
					print('Minibatch loss at step %d: %f' % (step, l))
            		print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            		print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# With dropout regularization

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
alpha = 0.15

graph = tf.Graph()

with graph.as_default():

	tf.set_random_seed(1)

	# Input data

	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size,
		image_size, num_channels))
	tf_train_labels  = tf.placeholder(tf.float32, shape+(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset  = tf.constant(test_dataset)

	# Decay learning rate
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(alpha, global_step, 100, 0.95,
		staircase=True)

	# Weights
	layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases  = tf.Variable(tf.zeros([depth]))
    
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases  = tf.Variable(tf.constant(1.0, shape=[depth]))

    layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases  = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases  = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
    def model(data):

    	conv_1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    	hidden_1 = tf.nn.relu(conv_1 + layer1_biases)
    	max_pool_1 = tf.nn.max_pool(hidden_1, ksize=[1, 2, 2, 1], 
    		strides=[1, 2, 2, 1], padding='SAME')