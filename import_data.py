import os
import requests
import gzip
import numpy as np


url_path = 'http://yann.lecun.com/exdb/mnist/'
folder   = os.getcwd()
#train-images-idx3-ubyte.gz	
#train-labels-idx1-ubyte.gz
#t10k-images-idx3-ubyte.gz
#t10k-labels-idx1-ubyte.gz

def load_mnist(mnist_dirname):

	if not os.path.exists(os.path.join(folder, mnist_dirname)):
		print('Creating datafolder..')
		os.makedirs(mnist_dirname) 

	print('Downloading data..')
	X_train = _load_mnist('train-images-idx3-ubyte.gz', mnist_dirname, 16).reshape((-1, 784))
	y_train = _load_mnist('train-labels-idx1-ubyte.gz', mnist_dirname, 8)
	X_test = _load_mnist('t10k-images-idx3-ubyte.gz', mnist_dirname, 16).reshape((-1, 784))
	y_test = _load_mnist('t10k-labels-idx1-ubyte.gz', mnist_dirname, 8)

	print('Completed!')

	return X_train, y_train, X_test, y_test

def download_mnist(filename, data_directory):

	url       = 'http://yann.lecun.com/exdb/mnist/' + filename
	directory = os.path.join(folder, data_directory)

	if not os.path.exists(os.path.join(directory, filename)):
		resp = requests.get(url)

		with open(os.path.join(directory,filename), 'wb') as file:
			file.write(resp.content)

	return os.path.join(directory, filename)

def _load_mnist(filename, data_directory, header_size):

	data_filepath = download_mnist(filename, data_directory)

	with gzip.open(data_filepath, 'rb') as fil:
		data = np.frombuffer(fil.read(), np.uint8, offset=header_size)
	return np.asarray(data, dtype=np.uint8)




X, y, l, k = load_mnist('data_mnist')

