from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
	# DL a file if not present, check size
	if not os.path.exists(filename):
		filename, _ = urlretrieve(url + filename, filename)
		statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified %s' % filename)
	else:
		print(statinfo.st_size)
		raise Exception('Failed to verify ' + filename + '.  Can you get it \
				with a browser?')
	return filename

filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
	f = zipfile.ZipFile(filename)
	for name in f.namelist():
		return tf.compat.as_str(f.read(name))
	f.close()

text = read_data(filename)
print('Data size %d' % len(text))

# Create validation set
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

# Utility functions to map char to vocab id's and back

vocabulary_size = len(string.ascii_lowercase) + 1 # a-z + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
	if char in string.ascii_lowercase:
		return ord(char) - first_letter + 1
	elif char == ' ':
		return 0
	else:
		print('Unexpected character: %s' % char)
		return 0

def id2char(dictid):
	if dictid > 0:
		return chr(dictid + first_letter - 1)
	else:
		return ' '

print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))

# Function to generate a training batch for the LSTM model

batch_size = 64
num_unrollings = 10

class BatchGenerator(object):
	def __init__(self, text, batch_size, num_unrollings):
		self._text = text
		self._text_size = len(text)
		self._batch_size = batch_size
		self._num_unrollings = num_unrollings
		segment = self._text_size // batch_size
		self._cursor = [offset * segment for offset in range(batch_size)]
		self._last_batch = self._next_batch()

	def _next_batch(self):
		# Generate a single batch from the current cursor position in the data
		batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
		for b in range(self._batch_size):
			batch[b, char2id(self._text[self._cursor[b]])] = 1.0
			self._cursor[b] = (self._cursor[b] + 1) % self._text_size
		return batch

	def next(self):
		# Generate next array of batches from data
		# Array consists of the last batch of the previous array
		# followed by num_unrollings new ones
		batches = [self._last_batch]
		for step in range(self._num_unrollings):
			batches.append(self._next_batch())
		self._last_batch = batches[-1]
		return batches

def characters(probabilities):
	# Turn a 1 hot encoding or a prob distro over the char back 
	# into its most likely char representation
	return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
	# Convert a sequence of batches back into their most likely 
	# string representation
	s = [''] * batches[0].shape[0]
	for b in batches:
		s = [''.join(x) for x in zip(s,characters(b))]
	return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))

# Show shape or content of variables to understand structure

print(train_batches.next()[1].shape)
print(len(train_text) // _batch_size)
print(len(string.ascii_lowercase))
print(np.zeros(shape=(2,4), dtype=np.float))

def logprob(predictions, labels):
	# Log prob of true labels in a predicted batch
	predictions[predictions < 1e-10] = 1e-10
	return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
	# Sample 1 element from distribution assumed to be array of normalized probs
	r = random.uniform(0, 1)
	s = 0
	for i in range(len(distribution)):
		s += distribution[i]
		if s >= r:
			return i
	return len(distribution) - 1

def sample(prediction):
	# Turn a (col) prediction into 1 hot encoded samples
	p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
	p[0, sample_distribution(prediction[0])] = 1.0
	return p

def random_distribution():
	# Generate random column of probabilities
	b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
	return b / np.sum(b,1)[:,None]

# Simple LSTM model

graph = tf.Graph()

with graph.as_default():

	tf.set_random_seed(1)

	# Parameters

	# INput gate: input, previous output, and bias