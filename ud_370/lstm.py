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
