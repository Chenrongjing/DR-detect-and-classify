import createDatabase as createDatabase
import numpy as np
import matplotlib.pyplot as pyplot

import theano
import theano.tensor as T

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum,adagrad
from lasagne.nonlinearities import leaky_rectify
from lasagne.regularization import regularize_layer_params
from lasagne.regularization import regularize_layer_params_weighted, l2, l1

import time

input_var = T.tensor4('inputs')
target_var = T.matrix('targets')

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def createCNN():
	network = lasagne.layers.InputLayer(shape=(None, 3, 512, 512), input_var=input_var)

	network=lasagne.layers.Conv2DLayer(
		network, num_filters=16, filter_size=(3, 3), stride=(1,1), pad=(1,1),
		nonlinearity=lasagne.nonlinearities.leaky_rectify,
		W=lasagne.init.GlorotUniform())
	network=lasagne.layers.BatchNormLayer(network)

	network=lasagne.layers.Conv2DLayer(
		network, num_filters=16, filter_size=(3, 3), stride=(1,1), pad=(1,1),
		nonlinearity=lasagne.nonlinearities.leaky_rectify,
		W=lasagne.init.GlorotUniform())
	network=lasagne.layers.BatchNormLayer(network)

	network=lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=(2,2))

	network=lasagne.layers.Conv2DLayer(
		network, num_filters=32, filter_size=(3, 3), stride=(1,1), pad=(1,1),
		nonlinearity=lasagne.nonlinearities.leaky_rectify,
		W=lasagne.init.GlorotUniform())
	network=lasagne.layers.BatchNormLayer(network)

	network=lasagne.layers.Conv2DLayer(
		network, num_filters=32, filter_size=(3, 3), stride=(1,1), pad=(1,1),
		nonlinearity=lasagne.nonlinearities.leaky_rectify,
		W=lasagne.init.GlorotUniform())
	network=lasagne.layers.BatchNormLayer(network)

	network=lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=(2,2))

	network=lasagne.layers.Conv2DLayer(
		network, num_filters=64, filter_size=(3, 3), stride=(1,1), pad=(1,1),
		nonlinearity=lasagne.nonlinearities.leaky_rectify,
		W=lasagne.init.GlorotUniform())
	network=lasagne.layers.BatchNormLayer(network)

	network=lasagne.layers.Conv2DLayer(
		network, num_filters=64, filter_size=(3, 3), stride=(1,1), pad=(1,1),
		nonlinearity=lasagne.nonlinearities.leaky_rectify,
		W=lasagne.init.GlorotUniform())
	network=lasagne.layers.BatchNormLayer(network)

	network=lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=(2,2))

	network=lasagne.layers.Conv2DLayer(
		network, num_filters=128, filter_size=(3, 3), stride=(1,1), pad=(1,1),
		nonlinearity=lasagne.nonlinearities.leaky_rectify,
		W=lasagne.init.GlorotUniform())
	network=lasagne.layers.BatchNormLayer(network)

	network=lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=(2,2))

	network=lasagne.layers.Conv2DLayer(
		network, num_filters=128, filter_size=(3, 3), stride=(1,1), pad=(1,1),
		nonlinearity=lasagne.nonlinearities.leaky_rectify,
		W=lasagne.init.GlorotUniform())
	network=lasagne.layers.BatchNormLayer(network)

	network=lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=(2,2))

	network=lasagne.layers.Conv2DLayer(
		network, num_filters=256, filter_size=(3, 3), stride=(1,1), pad=(1,1),
		nonlinearity=lasagne.nonlinearities.leaky_rectify,
		W=lasagne.init.GlorotUniform())
	network=lasagne.layers.BatchNormLayer(network)

	network=lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=(2,2))

	network=lasagne.layers.Conv2DLayer(
		network, num_filters=256, filter_size=(3, 3), stride=(1,1), pad=(1,1),
		nonlinearity=lasagne.nonlinearities.leaky_rectify,
		W=lasagne.init.GlorotUniform())
	network=lasagne.layers.BatchNormLayer(network)

	network=lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=(2,2))

	denseLayer1 = lasagne.layers.DenseLayer(
		lasagne.layers.dropout(network, p=.2),
		W=lasagne.init.GlorotUniform(),
		num_units=1024,
		nonlinearity=lasagne.nonlinearities.leaky_rectify)
	denseLayer2 = lasagne.layers.DenseLayer(
		lasagne.layers.dropout(denseLayer1, p=.2),
		W=lasagne.init.GlorotUniform(),
		num_units=1024,
		nonlinearity=lasagne.nonlinearities.leaky_rectify)
	network = lasagne.layers.DenseLayer(denseLayer2,
		W=lasagne.init.GlorotUniform(),
		num_units=4,
		nonlinearity=lasagne.nonlinearities.softmax)

	global l1penalty, l2penalty
	layers={denseLayer1: 1e-6, denseLayer2: 1e-6}
	l2penalty = lasagne.regularization.regularize_layer_params_weighted(layers, lasagne.regularization.l2)
	l1penalty = lasagne.regularization.regularize_layer_params(layers, l1) * 2e-6

	return network

databases = ['/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base11','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base12','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base13','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base14','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base21','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base22','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base23','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base24','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base31','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base32','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base33','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base34']

def main():

	network=createCNN()
	prediction = lasagne.layers.get_output(network, deterministic=True)
	params = lasagne.layers.get_all_params(network, trainable=True)
	predict_function = theano.function([input_var], prediction)

	with np.load('Networks/1024FC/DRNetwork129.npz') as f:
	     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)

	X=np.zeros((100,3,512,512))
	y=np.zeros((100,4))

	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
			                                        target_var)
	test_loss = test_loss.mean()

	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()
	loss += l1penalty + l2penalty

	acc = T.mean(T.eq(T.argmax(prediction, axis=1), T.argmax(target_var, axis=1)))

	test_fn = theano.function([input_var, target_var], [loss, acc])

	numBatches=20

	test_err = 0
	test_acc = 0
	test_batches = 0

	start_time = time.time()

	for index in range(0,12):
		print("Index:")
		print(index)
		X_test = np.memmap('Database/X'+str(index)+'.npz', dtype='float32', mode='r', shape=(100,3,512,512))
		y_test = np.memmap('Database/y'+str(index)+'.npz', dtype='float32', mode='r', shape=(100,4))

		print("Testing model...")

		# In each epoch, we do a full pass over the training data:
		for batch in iterate_minibatches(X_test, y_test, numBatches, shuffle=True):
			inputs, targets = batch
			err, acc = test_fn(inputs, targets)
			test_err += err
			test_acc += acc
			test_batches += 1

		#print("Predictions:")
		#print(np.argmax(predict_function(X),1))
		#print("Targets:")
		#print(np.argmax(y,1))
		

	# Then we print the results for this epoch:
	print("Epoch took {:.3f}s".format(time.time() - start_time))
	print("  testing loss:\t\t{:.12f}".format(test_err / test_batches))
	print("  testing accuracy:\t\t{:.4f} %").format(test_acc / test_batches * 100)
if __name__ == '__main__':
	main()
