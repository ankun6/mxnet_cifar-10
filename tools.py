import pickle
import numpy as np


def load_cifar10():
	data = []
	labels = []
	
	for num in range(1, 6):
		with open('./data/data_batch_' + str(num), 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
			data.append(np.array(dict[b'data']))
			labels.append(np.array(dict[b'labels']))
			fo.close()
	
	with open('./data/test_batch', 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		eval_data = np.array(dict[b'data']).reshape(10000, 3072)
		eval_labels = np.array(dict[b'labels']).reshape(10000)
		fo.close()
	
	train_data = np.array(data).reshape((50000, 3072))
	train_labels = np.array(labels).reshape(50000)
	
	return (train_data, train_labels), (eval_data, eval_labels)


def to4d(img):
	return img.reshape(img.shape[0], 3, 32, 32).astype(np.float32) / 255
