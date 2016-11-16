#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import sys
import os
import time
import random

import numpy as np
import theano
import theano.tensor as T

import Image
import ImageOps
from pandas import DataFrame


import lasagne

from sklearn import cross_validation

PIXELS = 227

# ############################ defineClassToPathDict #######################
# Here we take the classes as string arguments in order to add the negative
# ones and map each class (positive or negative) to its corresponding data
# path using the classToPathDict dictionary.

def defineClassToPathDict(classesPositive):
	
	for i in xrange(0,len(classesPositive)):
		if classesPositive[i] == 'voice_adults':
			classesPositive[i] = 'voice_(adults)'
		if classesPositive[i] == 'voice_children':
			classesPositive[i] = 'voice_(children)'
	
	initialPath = '/home/nikos/Desktop/NCSRDemokritos/CNNs-Soundscape-Quality-Estimation/data/complete_dataset'
	negativeClassString = 'not_'
	classesPositiveNegative = []
	for i in classesPositive:
		classesPositiveNegative.append(i + '/' + i)
		classesPositiveNegative.append(i + '/' + negativeClassString + i)
	classToPathDict= {}.fromkeys(classesPositiveNegative)
	for i in classToPathDict:
		classToPathDict[i] = initialPath + '/' + i
	
	return classToPathDict

# ############################ mapInstancesToMultipleLabels #######################
# Here we map each instance to its corresponding labels in a dictionary which has
# every instance as a key, while its value is a list of strings which are the 
# multiple labels for this key-instance

def mapInstancesToMultipleLabels(classToPathDict):
	
	pathToWavs = '/home/nikos/Desktop/NCSRDemokritos/CNNs-Soundscape-Quality-Estimation/Documents/soundscapeanalysis/wav2'
	#jpgInstances = [i+'.jpg' for i in os.listdir(pathToWavs)]
	pngInstances = [i+'.png' for i in os.listdir(pathToWavs)]
	png_rnoise21Instances = [i+'_rnoise21.png' for i in os.listdir(pathToWavs)]
	png_rnoise22Instances = [i+'_rnoise22.png' for i in os.listdir(pathToWavs)]
	png_rnoise23Instances = [i+'_rnoise23.png' for i in os.listdir(pathToWavs)]
	png_rnoise11Instances = [i+'_rnoise11.png' for i in os.listdir(pathToWavs)]
	png_rnoise12Instances = [i+'_rnoise12.png' for i in os.listdir(pathToWavs)]
	png_rnoise13Instances = [i+'_rnoise13.png' for i in os.listdir(pathToWavs)]
	png_rnoise01Instances = [i+'_rnoise01.png' for i in os.listdir(pathToWavs)]
	png_rnoise02Instances = [i+'_rnoise02.png' for i in os.listdir(pathToWavs)]
	png_rnoise03Instances = [i+'_rnoise03.png' for i in os.listdir(pathToWavs)]
	#jpg_02AInstances = [i+'_02A.jpg' for i in os.listdir(pathToWavs)]
	#jpg_02BInstances = [i+'_02B.jpg' for i in os.listdir(pathToWavs)]
	jpgInstances = pngInstances + png_rnoise21Instances + png_rnoise21Instances + png_rnoise22Instances + png_rnoise23Instances + png_rnoise11Instances + png_rnoise12Instances + png_rnoise13Instances + png_rnoise01Instances + png_rnoise02Instances + png_rnoise03Instances
	instancesToMultipleLabelsMap = {}.fromkeys(jpgInstances)

	instancesPerClass = []
	classNames = []
	for i in classToPathDict:
		classNames.append(i)
		instancesPerClass.append(os.listdir(classToPathDict[i]))
	
	for i,className in zip(instancesPerClass, classNames):
		for instance in instancesToMultipleLabelsMap:
			if instance in i:
				if(instancesToMultipleLabelsMap[instance] == None):
					instancesToMultipleLabelsMap[instance] = [className]
				else:
					instancesToMultipleLabelsMap[instance].append(className)
	
	return instancesToMultipleLabelsMap

# ############################ load_dataset #################################
# This function simply organizes the dataset for each binary classification
# problem. It returns the training, validation and testing set each time

def load_dataset(className, classToPathDict, instancesToMultipleLabelsMap, r_trn = 0.8, r_vld = 0.1, r_tst = 0.1, seed = 123):
	
	if className == 'voice_adults':
		className = 'voice_(adults)'
	if className == 'voice_children':
		className = 'voice_(children)'
		
	negativeClassString = 'not_'
	
	positiveClass = className + '/' + className
	negativeClass = className + '/' + negativeClassString + className
	
	X_N = []
	X_P = []
	labels = []
	instanceNames = []
		
	for i in instancesToMultipleLabelsMap:
		if not instancesToMultipleLabelsMap[i]:
			continue
		if positiveClass in instancesToMultipleLabelsMap[i]:
			img = Image.open(classToPathDict[positiveClass] +'/'+ i)
			img = np.asarray(img, dtype='float32') / 255
			img = img.transpose(2,0,1).reshape(4,PIXELS,PIXELS)
			X_P.append(img)
			instanceNames.append(i)
			labels.append(1)
		if negativeClass in instancesToMultipleLabelsMap[i]:
			img = Image.open(classToPathDict[negativeClass] +'/'+ i)
			img = np.asarray(img, dtype='float32') / 255
			img = img.transpose(2,0,1).reshape(4,PIXELS,PIXELS)
			X_N.append(img)
			instanceNames.append(i)
			labels.append(0)
	
	X = X_N + X_P
	c = list(zip(X, labels, instanceNames))
	random.shuffle(c)
	
	inputs = np.array(X)
	
	labels = np.asarray(labels, dtype = np.int8)
	labels = labels.reshape(labels.shape[0],1)
	
	instanceNames = np.asarray(instanceNames, dtype = np.str_)
	
	if seed == None:
	    seed = random.randint(0,10000)
	    print(seed)
	
	# Stratified dataset split into training, validation and testing
	trn, vld, tst = [],[],[]
	
	for clss in np.unique(labels):
		# Create list of indices for class
		ndxs = [i for i, x in enumerate(labels) if x == clss]
		
		# Compute number of samples for each subset
		ntst = int(round(len(ndxs)*r_tst))
		nvld = int(round(len(ndxs)*r_vld))
		ntrn = len(ndxs) - ntst - nvld
		
		# Randomly select class indices for each subset
		
		random.seed(seed)
		trn.extend([ndxs.pop(x) for x in sorted(random.sample(range(0,len(ndxs)), ntrn),reverse=True)])
		random.seed(seed)
		vld.extend([ndxs.pop(x) for x in sorted(random.sample(range(0,len(ndxs)), nvld),reverse=True)])
		random.seed(seed)
		tst.extend([ndxs.pop(x) for x in sorted(random.sample(range(0,len(ndxs)), ntst),reverse=True)])
		
	# Random shuffle the resulting vectors of indices
	[random.shuffle(x) for x in [trn, vld, tst]]
		
	# Return results
	return inputs[trn], labels[trn], inputs[vld], labels[vld], inputs[tst], labels[tst], instanceNames[trn], instanceNames[vld], instanceNames[tst]
	
def build_simple(input_var=None):
	network = lasagne.layers.InputLayer(shape=(None, 3, 227, 227), input_var=input_var)
	
	network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=100, nonlinearity=lasagne.nonlinearities.rectify)
	
	network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=1,nonlinearity=lasagne.nonlinearities.sigmoid)

def build_cnnrnn(input_var=None):
	
    network = lasagne.layers.InputLayer(shape=(None, 3, 227, 227), input_var=input_var)
                                        
    network = lasagne.layers.Conv2DLayer(network, num_filters=96, filter_size=(7,7), pad=0, flip_filters=False,stride=2,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Pool2DLayer(network, pool_size = (3,3), stride = 2, mode = 'max')
    network = lasagne.layers.LocalResponseNormalization2DLayer(network, alpha = 0.0001, beta = 0.75, n = 5)
    network = lasagne.layers.Conv2DLayer(network, num_filters = 384, filter_size = (5,5), pad = 0, flip_filters=False,stride=2,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Pool2DLayer(network, pool_size = (3,3), stride = 2, mode = 'max')
    network = lasagne.layers.LocalResponseNormalization2DLayer(network, alpha = 0.0001, beta = 0.75, n = 5)
    network = lasagne.layers.Conv2DLayer(network, num_filters=512, filter_size=(3,3), pad=1, flip_filters=False,stride=2,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(network, num_filters=512, filter_size=(3,3), pad=1, flip_filters=False,stride=2,nonlinearity=lasagne.nonlinearities.rectify)
    #network = lasagne.layers.Conv2DLayer(network, num_filters=384, filter_size=(3,3), pad=1, flip_filters=False,stride=2,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Pool2DLayer(network, pool_size = (3,3), stride = 2, mode = 'max')
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=4096, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=101, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.0), num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
    
	
    return network
    
def build_cnn(input_var=None):
    #  As a model we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, will be 3 dimension (since we have spectrograms as inputs),
    # while the second and third will be the total bytes of the image:
    network = lasagne.layers.InputLayer(shape=(None, 4, 227, 227),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    # we changed that to 3x3 (Thodoris)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 1-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    return network
    
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, instanceNames, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], instanceNames[excerpt]
        
def compute_recall_precision(labels, preds):
	
	TruePositives = 0
	TrueNegatives = 0
	FalseNegatives = 0
	FalsePositives = 0
	
	for label, pred in zip(labels, preds):
		if int(pred) == 1 and label[0] == 1:
			TruePositives += 1
		elif int(pred) == 0 and label[0] == 0:
			TrueNegatives += 1
		elif int(pred) == 0 and label[0] == 1:
			FalseNegatives += 1
		elif int(pred) == 1 and label[0] == 0:
			FalsePositives += 1
	
	if(TruePositives == 0 and FalsePositives == 0):
		recall = 0
	else:
		recall = TruePositives / (TruePositives + FalsePositives)
	if(TruePositives == 0 and FalseNegatives == 0):
		precision = 0
	else:
		precision = TruePositives / (TruePositives + FalseNegatives)
	return recall, precision
	
def multilabel_fn_gather_results(preds, instanceNames, instancesToMultipleLabelsMap, multiLabelClassificationResults, c):
	
	negativeClass = c + '/' + 'not_' + c
	positiveClass = c + '/' + c
	
	instanceNamesTemp = [i for i in instanceNames]
	#targetsTemp = [i for i in targets]
	
	TruePositives = 0
	TrueNegatives = 0
	FalsePositives = 0
	FalseNegatives = 0
	
	if multiLabelClassificationResults == {}:
		multiLabelClassificationResults= {}.fromkeys(instanceNamesTemp)
		
		for instance, pred  in zip(instanceNamesTemp, preds):
			if positiveClass in instancesToMultipleLabelsMap[instance] and int(pred) == 1:
				multiLabelClassificationResults[instance] = [[1, 1]]
			elif positiveClass in instancesToMultipleLabelsMap[instance] and int(pred) == 0:
				multiLabelClassificationResults[instance] = [[1, 0]]
			elif negativeClass in instancesToMultipleLabelsMap[instance] and int(pred) == 0:
				multiLabelClassificationResults[instance] = [[0, 0]]
			elif negativeClass in instancesToMultipleLabelsMap[instance] and int(pred) == 1:
				multiLabelClassificationResults[instance] = [[0, 1]]
	else:
		for instance, pred in zip(instanceNamesTemp, pred):
			if positiveClass in instancesToMultipleLabelsMap[instance] and int(pred) == 1:
				multiLabelClassificationResults[instance].append([1, 1])
			elif positiveClass in instancesToMultipleLabelsMap[instance] and int(pred) == 0:
				multiLabelClassificationResults[instance].append([1, 0])
			elif negativeClass in instancesToMultipleLabelsMap[instance] and int(pred) == 0:
				multiLabelClassificationResults[instance].append([0, 0])
			elif negativeClass in instancesToMultipleLabelsMap[instance] and int(pred) == 1:
				multiLabelClassificationResults[instance].append([0, 1])
				
	return multiLabelClassificationResults

def multilabel_fn_compute_metrics(multiLabelClassificationResults):
	
	rec = 0
	pres = 0
	count = 0
	
	for instance in multiLabelClassificationResults:
		Truth = 0
		Prediction = 0
		TruthAndPrediction = 0
		for TruthPredictionPair in multiLabelClassificationResults[instance]:
			if TruthPredictionPair[0] == 1:
				Truth += 1
			if TruthPredictionPair[1] == 1:
				Prediction += 1
			if TruthPredictionPair[0] == TruthPredictionPair[1]:
				TruthAndPrediction += 1
		
		if(Truth == 0):
			rec += 0
		else:
			rec += TruthAndPrediction / Truth
			
		if(Prediction == 0):
			pres += 0
		else:
			pres += TruthAndPredition / Prediction
		
		count += 1
	
	pres = pres / count
	rec = rec / count
	
	return pres, rec
	
# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
def main(model='cnn', num_epochs=500, classes = ['vehicles']):
	
	# Defining dataset paths
	
    classToPathDict = defineClassToPathDict(classes)
    
    instancesToMultipleLabelsMap = mapInstancesToMultipleLabels(classToPathDict)
    multiLabelClassificationResults = {}
    
    # Prepare Data
    for c in classes:
		X_train, y_train, X_val, y_val, X_tst, y_tst, instanceNames_train, instanceNames_val, instanceNames_tst = load_dataset(c, classToPathDict, instancesToMultipleLabelsMap)
		
		# Build neural network model 
		print("Building network ...")

		
		# Prepare Theano variables for targets
		input_var = theano.tensor.tensor4('inputs')
		target_var = theano.tensor.imatrix('targets')

		# Create neural network model (depending on first command line parameter)
		print("Building model and compiling functions...")
		
		if model == 'cnn':
			network = build_cnn(input_var)
		elif model == 'cnn+rnn':
			network = build_cnnrnn(input_var)
		elif model == 'simple':
			network = build_simple(input_var)

		# Create a loss expression for training, i.e., a scalar objective we want
		# to minimize (for our multi-class problem, it is the cross-entropy loss):
		prediction = lasagne.layers.get_output(network)
		loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
		loss = loss.mean()
		# We could add some weight decay as well here, see lasagne.regularization.

		# Create update expressions for training, i.e., how to modify the
		# parameters at each training step. Here, we'll use Stochastic Gradient
		# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		params = lasagne.layers.get_all_params(network, trainable=True)
		updates = lasagne.updates.nesterov_momentum(
				loss, params, learning_rate=0.02, momentum=0.5)

		# Create a loss expression for validation/testing. The crucial difference
		# here is that we do a deterministic forward pass through the network,
		# disabling dropout layers.
		test_prediction = lasagne.layers.get_output(network, deterministic=True)
		test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
																target_var)
		test_loss = test_loss.mean()
		# As a bonus, also create an expression for the classification accuracy:
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
						  dtype=theano.config.floatX)

		# Compile a function performing a training step on a mini-batch (by giving
		# the updates dictionary) and returning the corresponding training loss:
		train_fn = theano.function([input_var, target_var], loss, updates=updates)

		# Compile a second function computing the validation loss and accuracy:
		val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])
		
		# Finally, launch the training loop.
		print("Starting training...")
		# We iterate over epochs:
		for epoch in range(num_epochs):
			# In each epoch, we do a full pass over the training data:
			train_err = 0
			train_batches = 0
			start_time = time.time()
			for batch in iterate_minibatches(X_train, y_train, instanceNames_train, 10, shuffle=True):
				inputs, targets, instanceNamesTemp = batch
				train_err += train_fn(inputs, targets)
				train_batches += 1

			# And a full pass over the validation data:
			val_err = 0
			val_acc = 0
			val_rec = 0
			val_pres = 0
			val_batches = 0
			preds = []
			true = 0
			count = 0
			for batch in iterate_minibatches(X_val, y_val, instanceNames_val, 1, shuffle=False):
				
				inputs, targets, instanceNamesTemp = batch
				err, acc, temp_preds = val_fn(inputs, targets)
				val_err += err
				val_acc += acc
				
				#print(temp_preds)
				#print(targets)
				if(temp_preds > 0.5 and targets == 1):
					true += 1
				elif(temp_preds < 0.5 and targets == 0):
					true += 1
				count += 1
				preds.append(temp_preds[0][0].tolist())

				val_batches += 1

			val_recall, val_precision = compute_recall_precision(y_val, preds)
			print("printing the alternative accuracy")
			print(true/count)
			# Then we print the results for this epoch:
			print("Epoch {} of {} took {:.3f}s".format(
				epoch + 1, num_epochs, time.time() - start_time))
			print("   Binary Classification : training loss:\t\t{:.6f}".format(train_err / train_batches))
			print("   Binary Classification : validation loss:\t\t{:.6f}".format(val_err / val_batches))
			print("   Binary Classification : validation accuracy:\t\t{:.2f} %".format(
				val_acc / val_batches * 100))
			print("   Binary Classification : validation recall:\t\t{:.2f} %".format(
				val_recall * 100))
			print("   Binary Classification : validation precision:\t{:.2f} %".format(
				val_precision * 100))

		# After training, we compute and print the test metrics:
		tst_err = 0
		tst_acc = 0
		tst_rec = 0
		tst_pres = 0
		tst_batches = 0
		preds = []
		for batch in iterate_minibatches(X_tst, y_tst, instanceNames_tst, 1, shuffle=False):
			
			inputs, targets, instanceNamesTemp = batch
			err, acc, temp_preds = val_fn(inputs, targets)
			tst_err += err
			tst_acc += acc
			#print(temp_preds)
			#print(targets)
			
			preds.append(temp_preds[0][0].tolist())
			
			tst_batches += 1
		
		tst_recall, tst_precision = compute_recall_precision(y_tst, preds)
		#val_fn_more_metrics(preds, instanceNames_tst, instancesToMultipleLabelsMap, c)
		
		multiLabelClassificationResults = multilabel_fn_gather_results(preds, instanceNames_tst, instancesToMultipleLabelsMap, multiLabelClassificationResults, c)
		
		print("Final results:")
		print("   Binary Classification : test loss:\t\t\t{:.6f}".format(tst_err / tst_batches))
		print("   Binary Classification : test accuracy:\t\t{:.2f} %".format(
			tst_acc / tst_batches * 100))
		print("   Binary Classification : test recall:\t\t{:.2f} %".format(
				tst_recall * 100))
		print("   Binary Classification : test precision:\t\t{:.2f} %".format(
			tst_precision * 100))
			
		# Optionally, you could now dump the network weights to a file like this:
		# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
		#
		# And load them again later on like this:
		# with np.load('model.npz') as f:
		#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		# lasagne.layers.set_all_param_values(network, param_values)
			
    multi_rec, multi_pres = multilabel_fn_compute_metrics(multiLabelClassificationResults)
    print("   Multilabel Classification : test recall:\t\t{:.2f} %".format(multi_rec * 100))
    print("   Multilabel Classification : test precision:\t\t{:.2f} %".format(multi_pres * 100))
	
	
    
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s MODEL EPOCHS CLASSES" % sys.argv[0])
        print()
        print("MODEL: 'cnn+rnn' for using the customized Long-term Recurrent Convolutional Network (not implemented yet),")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
        print("CLASSES: the name of each class for our multi-label problem")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        if len(sys.argv) > 3:
			kwargs['classes'] = sys.argv[3:]
        main(**kwargs)
