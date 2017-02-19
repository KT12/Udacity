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

