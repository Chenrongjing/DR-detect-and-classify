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
import sys

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
		lasagne.layers.dropout(network, p=.3),
		W=lasagne.init.GlorotUniform(),
		num_units=1024,
		nonlinearity=lasagne.nonlinearities.leaky_rectify)
	denseLayer2 = lasagne.layers.DenseLayer(
		lasagne.layers.dropout(denseLayer1, p=.3),
		W=lasagne.init.GlorotUniform(),
		num_units=1024,
		nonlinearity=lasagne.nonlinearities.sigmoid)
	denseLayer3 = lasagne.layers.DenseLayer(
		lasagne.layers.dropout(denseLayer2, p=.3),
		W=lasagne.init.GlorotUniform(),
		num_units=1024,
		nonlinearity=lasagne.nonlinearities.sigmoid)
	network = lasagne.layers.DenseLayer(denseLayer3,
		W=lasagne.init.GlorotUniform(),
		num_units=4,
		nonlinearity=lasagne.nonlinearities.softmax)

	global l1penalty, l2penalty
	layers={denseLayer1: 1e-6, denseLayer2: 1e-6, denseLayer3: 1e-6, network: 1e-6}
	l2penalty = lasagne.regularization.regularize_layer_params_weighted(layers, lasagne.regularization.l2)
	l1penalty = lasagne.regularization.regularize_layer_params(layers, l1) * 1e-7

	return network

databases = ['/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base11','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base12','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base13','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base14','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base21','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base22','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base23','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base24','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base31','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base32','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base33','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base34']

def main():

	network=createCNN()
	outputDirectory=''

	trainingLoss=np.float32([])
	trainingPrecision=np.float32([])
	validationLoss=np.float32([])
	validationPrecision=np.float32([])
	
	if(len(sys.argv)>=2):		
		print('Number of Epochs: '+str(int(sys.argv[1])))
		num_epochs=int(sys.argv[1])
	
	if(len(sys.argv)>=3):
		print('Output Directory: '+str(sys.argv[2]))
		outputDirectory=sys.argv[2]

	if(len(sys.argv)>=4):
		if(sys.argv[3]!='0'):
			print('Network: '+str(sys.argv[3]))
			with np.load(sys.argv[3]) as f:
			    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
			lasagne.layers.set_all_param_values(network, param_values)
			start=int(sys.argv[3][sys.argv[3].index('DRNetwork')+len('DRNetwork'):sys.argv[3].index('.npz')])
			print('Start: '+sys.argv[3])

			trainingLoss = np.memmap(outputDirectory+'/trainingLoss.dat', dtype='float32', mode='r')[0:start-1]
			trainingPrecision = np.memmap(outputDirectory+'/trainingPrecision.dat', dtype='float32', mode='r')[0:start-1]
			validationLoss = np.memmap(outputDirectory+'/validationLoss.dat', dtype='float32', mode='r')[0:start-1]
			validationPrecision = np.memmap(outputDirectory+'/validationPrecision.dat', dtype='float32', mode='r')[0:start-1]
		else:
			start=0
			print('Start: '+sys.argv[3])
	learningRate=0.00005
	if(len(sys.argv)>=5):
		learningRate=np.float32(sys.argv[4])
	
	print('Learning Rate: '+str(learningRate))

	train_prediction = lasagne.layers.get_output(network, deterministic=True)
	train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, target_var)
	train_loss = train_loss.mean()
	train_loss += l1penalty + l2penalty

	train_acc = T.mean(T.eq(T.argmax(train_prediction, axis=1), T.argmax(target_var, axis=1)))

	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.adamax(train_loss, params, learning_rate=learningRate, beta1=0.9, beta2=0.999, epsilon=1e-08)

	validation_prediction = lasagne.layers.get_output(network, deterministic=True)
	validation_loss = lasagne.objectives.categorical_crossentropy(validation_prediction,target_var)
	validation_loss = validation_loss.mean()
	validation_loss += l1penalty + l2penalty

	validation_acc = T.mean(T.eq(T.argmax(validation_prediction, axis=1), target_var), dtype=theano.config.floatX)

	train_fn = theano.function([input_var, target_var], [train_loss, train_acc], updates=updates)
	val_fn = theano.function([input_var, target_var], [validation_loss, validation_acc])
	
	numBatches=20

	epoch = 0
	
	X=np.zeros((100,3,512,512))
	y=np.zeros((100,4))

	while (epoch<num_epochs):
		train_err = 0
		train_acc = 0
		train_batches = 0

		val_err = 0
		val_acc = 0
		val_batches = 0

		start_time = time.time()

		for index in range(0,12):
			print("Index:")
			print(index)
			X = np.memmap('Database/X'+str(index)+'.npz', dtype='float32', mode='r', shape=(100,3,512,512))
			y = np.memmap('Database/y'+str(index)+'.npz', dtype='float32', mode='r', shape=(100,4))		
			shuffledTrainIndices=np.arange(0,90)
			np.random.shuffle(shuffledTrainIndices)
			X_train=X[shuffledTrainIndices,:,:,:]
			y_train=y[shuffledTrainIndices]

			shuffledValidationIndices=np.arange(90,100)
			np.random.shuffle(shuffledValidationIndices)
			X_val=X[shuffledValidationIndices,:,:,:]
			y_val=y[shuffledValidationIndices]

			print("Training model...")

			# In each database, we do a full pass over the training data:
			for batch in iterate_minibatches(X_train, y_train, numBatches, shuffle=True):
				inputs, targets = batch
				err, acc = train_fn(inputs, targets)
				train_err += err
				train_acc += acc
				train_batches += 1

			# And a full pass over the validation data:
			for batch in iterate_minibatches(X_val, y_val, 4, shuffle=False):
				inputs, targets = batch
				err, acc = val_fn(inputs, targets)
				val_err += err
				val_acc += acc
				val_batches += 1

			#print("Predictions:")
			#print(np.argmax(predict_function(X),1))
			#print("Targets:")
			#print(np.argmax(y,1))
		
		trainingLoss=np.append(trainingLoss,np.float32(train_err / train_batches))
		trainingPrecision=np.append(trainingPrecision,np.float32(train_acc / train_batches * 100))
		validationLoss=np.append(validationLoss,np.float32(val_err / val_batches))
		validationPrecision=np.append(validationPrecision,np.float32(val_acc / val_batches * 100))		

		# Then we print the results for this epoch:
		print("Epoch {} took {:.3f}s".format(
		start+epoch+1, time.time() - start_time))
		print("  training loss:\t\t{:.12f}".format(train_err / train_batches))
		print("  training precision:\t\t{:.4f} %").format(train_acc / train_batches * 100)
		print("  validation loss:\t\t{:.12f}".format(val_err / val_batches))
		print("  validation precision:\t\t{:.4f} %".format(val_acc / val_batches * 100))
	
		np.savez(outputDirectory+'/DRNetwork'+str(start+epoch+1), *lasagne.layers.get_all_param_values(network))

		fp = np.memmap(outputDirectory+'/trainingLoss.dat', dtype='float32', mode='w+',shape=(np.size(trainingLoss)))
		fp[:]=trainingLoss
		fp = np.memmap(outputDirectory+'/trainingPrecision.dat', dtype='float32', mode='w+',shape=(np.size(trainingPrecision)))
		fp[:]=trainingPrecision
		fp = np.memmap(outputDirectory+'/validationLoss.dat', dtype='float32', mode='w+',shape=(np.size(validationLoss)))
		fp[:]=validationLoss
		fp = np.memmap(outputDirectory+'/validationPrecision.dat', dtype='float32', mode='w+',shape=(np.size(validationPrecision)))
		fp[:]=validationPrecision

		epoch=epoch+1

if __name__ == '__main__':
	main()
