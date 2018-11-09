import mxnet as mx
import numpy as np
import logging
import pickle

logging.getLogger().setLevel(logging.DEBUG)


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


epoch = 60
batch_size = 32


# 搭建网络
data = mx.sym.Variable('data')
conv1 = mx.sym.Convolution(data=data, num_filter=32, kernel=(3, 3), name='conv1')
act1  = mx.sym.Activation(data=conv1, act_type='relu', name='act1')
conv2 = mx.sym.Convolution(data=act1, num_filter=32, kernel=(3, 3), name='conv2')
act2  = mx.sym.Activation(data=conv1, act_type='relu', name='act2')
pool1 = mx.sym.Pooling(data=act2, kernel=(2, 2), pool_type='max', name='pool1')
drop1 = mx.sym.Dropout(data=pool1, p=0.25, name='drop1')
conv3 = mx.sym.Convolution(data=data, num_filter=64, kernel=(3, 3), name='conv3')
act3  = mx.sym.Activation(data=conv3, act_type='relu', name='act3')
conv4 = mx.sym.Convolution(data=act3, num_filter=64, kernel=(3, 3), name='conv4')
act4  = mx.sym.Activation(data=conv4, act_type='relu', name='act4')
pool2 = mx.sym.Pooling(data=act4, kernel=(2, 2), pool_type='max', name='pool2')
drop2 = mx.sym.Dropout(data=pool2, p=0.25, name='drop2')
flat  = mx.sym.Flatten(data=drop2, name='faltten')
fc1   = mx.sym.FullyConnected(data=flat, num_hidden=512, name='fc1')
drop3 = mx.sym.Dropout(data=fc1, p=0.5, name='drop3')
lenet = mx.sym.SoftmaxOutput(data=drop3, name='softmax')

# 创建模型
model = mx.mod.Module(symbol=lenet, context=mx.gpu())

# 加载测试数据
(train_data, train_labels), (eval_data, eval_labels) = load_cifar10()
train_iter = mx.io.NDArrayIter(to4d(train_data), train_labels, batch_size=batch_size, shuffle=True)
eval_iter  = mx.io.NDArrayIter(to4d(eval_data), eval_labels, batch_size=batch_size, shuffle=False)


# 训练
model.fit(train_data=train_iter, eval_data=eval_iter, eval_metric='acc', optimizer='sgd', num_epoch=epoch,
			optimizer_params={'learning_rate':0.0001, 'momentum': 0.7},
			batch_end_callback=mx.callback.Speedometer(batch_size=batch_size, frequent=batch_size))


# 保存模型
model.save_checkpoint('cifar10', epoch=epoch)

# 评估模型精度
score = model.score(eval_iter, ['acc'])
print(score)
