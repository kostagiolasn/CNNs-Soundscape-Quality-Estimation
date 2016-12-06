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

import csv


import lasagne

from sklearn import cross_validation
from sklearn import metrics

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
	
	print(classToPathDict)
	return classToPathDict

# ############################ mapInstancesToMultipleLabels #######################
# Here we map each instance to its corresponding labels in a dictionary which has
# every instance as a key, while its value is a list of strings which are the 
# multiple labels for this key-instance

def mapInstancesToMultipleLabels(classToPathDict):
	
	pathToWavs = '/home/nikos/Desktop/NCSRDemokritos/CNNs-Soundscape-Quality-Estimation/data/SoundscapeWAVs'
	#jpgInstances = [i+'.jpg' for i in os.listdir(pathToWavs)]
	pngInstances = [i[:-4]+'.png' for i in os.listdir(pathToWavs)]
	png_rnoise21Instances = [i[:-4]+'_rnoise21.png' for i in os.listdir(pathToWavs)]
	png_rnoise22Instances = [i[:-4]+'_rnoise22.png' for i in os.listdir(pathToWavs)]
	png_rnoise23Instances = [i[:-4]+'_rnoise23.png' for i in os.listdir(pathToWavs)]
	png_rnoise11Instances = [i[:-4]+'_rnoise11.png' for i in os.listdir(pathToWavs)]
	png_rnoise12Instances = [i[:-4]+'_rnoise12.png' for i in os.listdir(pathToWavs)]
	png_rnoise13Instances = [i[:-4]+'_rnoise13.png' for i in os.listdir(pathToWavs)]
	png_rnoise01Instances = [i[:-4]+'_rnoise01.png' for i in os.listdir(pathToWavs)]
	png_rnoise02Instances = [i[:-4]+'_rnoise02.png' for i in os.listdir(pathToWavs)]
	png_rnoise03Instances = [i[:-4]+'_rnoise03.png' for i in os.listdir(pathToWavs)]
	#jpg_02AInstances = [i+'_02A.jpg' for i in os.listdir(pathToWavs)]
	#jpg_02BInstances = [i+'_02B.jpg' for i in os.listdir(pathToWavs)]
	jpgInstances = pngInstances + png_rnoise21Instances + png_rnoise21Instances + png_rnoise22Instances + png_rnoise23Instances + png_rnoise11Instances + png_rnoise12Instances + png_rnoise13Instances + png_rnoise01Instances + png_rnoise02Instances + png_rnoise03Instances
	instancesToMultipleLabelsMap = {}.fromkeys(jpgInstances)

	instancesPerClass = []
	classNames = []
	print(classToPathDict)
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
	
	#print(instancesToMultipleLabelsMap)
	
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
	network = lasagne.layers.InputLayer(shape=(None, 4, 227, 227), input_var=input_var)
	
	network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=100, nonlinearity=lasagne.nonlinearities.rectify)
	
	network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5), num_units=1,nonlinearity=lasagne.nonlinearities.sigmoid)
	
	return network

def build_cnnrnn(input_var=None):
	
    network = lasagne.layers.InputLayer(shape=(None, 4, 227, 227), input_var=input_var)
                                        
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
        
def compute_recall_precision(labels, preds, threshold = 0.5):
	
	TruePositives = 0
	TrueNegatives = 0
	FalseNegatives = 0
	FalsePositives = 0
	
	for label, pred in zip(labels, preds):
		if pred >= threshold and label[0] == 1:
			TruePositives += 1
		elif pred < threshold and label[0] == 0:
			TrueNegatives += 1
		elif pred < threshold and label[0] == 1:
			FalseNegatives += 1
		elif pred >= threshold and label[0] == 0:
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
	
def multilabel_fn_gather_results(preds, instanceNames, instancesToMultipleLabelsMap, multiLabelClassificationResults, c, threshold = 0.5):
	
	negativeClass = c + '/' + 'not_' + c
	positiveClass = c + '/' + c
	
	instanceNamesTemp = [i for i in instanceNames]
	#targetsTemp = [i for i in targets]
	
	TruePositives = 0
	TrueNegatives = 0
	FalsePositives = 0
	FalseNegatives = 0
	#print(multiLabelClassificationResults)
	if multiLabelClassificationResults == {}:
		multiLabelClassificationResults= {}.fromkeys(instanceNamesTemp)
		
		for instance, pred  in zip(instanceNamesTemp, preds):
			if positiveClass in instancesToMultipleLabelsMap[instance] and pred >= threshold:
				multiLabelClassificationResults[instance] = [[1, 1]]
				TruePositives = TruePositives + 1
			elif positiveClass in instancesToMultipleLabelsMap[instance] and pred < threshold:
				multiLabelClassificationResults[instance] = [[1, 0]]
				FalseNegatives = FalseNegatives + 1
			elif negativeClass in instancesToMultipleLabelsMap[instance] and pred < threshold:
				multiLabelClassificationResults[instance] = [[0, 0]]
				TrueNegatives = TrueNegatives + 1
			elif negativeClass in instancesToMultipleLabelsMap[instance] and pred >= threshold:
				multiLabelClassificationResults[instance] = [[0, 1]]
				FalsePositives = FalsePositives + 1
		return multiLabelClassificationResults
	else:
		temp_multiLabelClassificationResults = {}.fromkeys(instanceNamesTemp)
		new_multiLabelClassificationResults = multiLabelClassificationResults.copy()
		new_multiLabelClassificationResults.update(temp_multiLabelClassificationResults) 
		for instance, pred in zip(instanceNamesTemp, preds):
			if positiveClass in instancesToMultipleLabelsMap[instance] and pred >= threshold:
				if not new_multiLabelClassificationResults[instance]:
					new_multiLabelClassificationResults[instance] = [[1, 1]]
				else:
					new_multiLabelClassificationResults[instance].append([1, 1])
			elif positiveClass in instancesToMultipleLabelsMap[instance] and pred < threshold:
				if not new_multiLabelClassificationResults[instance]:
					new_multiLabelClassificationResults[instance] = [[1, 0]]
				else:
					new_multiLabelClassificationResults[instance].append([1, 0])
			elif negativeClass in instancesToMultipleLabelsMap[instance] and pred < threshold:
				if not new_multiLabelClassificationResults[instance]:
					new_multiLabelClassificationResults[instance] = [[0, 0]]
				else:
					new_multiLabelClassificationResults[instance].append([0, 0])
			elif negativeClass in instancesToMultipleLabelsMap[instance] and pred >= threshold:
				if not new_multiLabelClassificationResults[instance]:
					new_multiLabelClassificationResults[instance] = [[0, 1]]
				else:
					new_multiLabelClassificationResults[instance].append([0, 1])
					
		return new_multiLabelClassificationResults

def multilabel_fn_compute_metrics(multiLabelClassificationResults):
	
	rec = 0
	pres = 0
	count = 0
	Truth = 0
	Prediction = 0
	TruthAndPrediction = 0
	
	
	for instance in multiLabelClassificationResults:

		for TruthPredictionPair in multiLabelClassificationResults[instance]:
			#print(TruthPredictionPair)
			if TruthPredictionPair[0] == 1:
				#print("truth")
				Truth += 1
			if TruthPredictionPair[1] == 1:
				Prediction += 1
				#print("prediction")
			if TruthPredictionPair[0] == 1 and TruthPredictionPair[1] == 1:
				TruthAndPrediction += 1
				#print("both")
		if(Truth == 0):
			rec += 0.0
		else:
			rec += TruthAndPrediction / Truth
			#print(rec)
			
		if(Prediction == 0):
			pres += 0.0
		else:
			pres += TruthAndPrediction / Prediction
			#print(pres)
		
		count += 1

	#pres = TruthAndPrediction / Prediction / count
	#rec = TruthAndPrediction / Truth / count
	pres = pres / count
	rec = rec / count
	f_score = 2.0 * pres * rec / (pres+rec)
	
	return pres, rec, f_score
	
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
		#updates = lasagne.updates.nesterov_momentum(
		#		loss, params, learning_rate=0.0001, momentum=0.5)
		
		updates = lasagne.updates.sgd(loss, params, learning_rate = 0.02)
		#updates = lasagne.updates.adam(loss, params)

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
		
		# We define a threshold
		threshold = 0.4
		
		# We iterate over epochs:
		for epoch in range(num_epochs):
			# In each epoch, we do a full pass over the training data:
			train_err = 0
			train_batches = 0
			start_time = time.time()
			#print(y_train)
			for batch in iterate_minibatches(X_train, y_train, instanceNames_train, 32, shuffle=True):
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
			for batch in iterate_minibatches(X_val, y_val, instanceNames_val, 1, shuffle=True):
				
				inputs, targets, instanceNamesTemp = batch
				err, acc, temp_preds = val_fn(inputs, targets)
				val_err += err
				val_acc += acc
				
				#print(temp_preds)
				#print(targets)
				if(temp_preds >= threshold and targets == 1):
					true += 1
				elif(temp_preds < threshold and targets == 0):
					true += 1
				count += 1
				preds.append(temp_preds[0][0].tolist())

				val_batches += 1

			val_recall, val_precision = compute_recall_precision(y_val, preds, threshold)
							
			if epoch == num_epochs-1:
				dataCSV_0 = open('predLabelsNotVehicles-Validation.csv', 'wb')
				dataCSV_1 = open('predLabelsVehicles-Validation.csv', 'wb')
				writer_0 = csv.writer(dataCSV_0, delimiter=',')
				writer_1 = csv.writer(dataCSV_1, delimiter=',')
				writer_0.writerow(['Predicted Label'])
				writer_1.writerow(['Predicted Label'])
				for label, pred_label in zip(y_val, preds):
					if label[0] == 0:
						data = [pred_label]
						writer_0.writerow(data)		
					else:
						data = [pred_label]
						writer_1.writerow(data)	
			"""else:
				dataCSV = open('test.csv', 'a')
				writer = csv.writer(dataCSV, delimiter=',')
				for label, pred_label in zip(y_val, preds):
					data = [epoch, y_val, preds]
					writer.writerow(data)"""
			
			if(val_precision + val_recall == 0):
				val_f_score = float('nan')
			else:
				val_f_score = 2.0*val_precision*val_recall/ float(val_precision+val_recall)
			
			if epoch == 0:
				dataCSVAcc = open('ValidationMetrics.csv', 'wb')
				writer_Acc = csv.writer(dataCSVAcc, delimiter=',')
				writer_Acc.writerow(['Epoch', 'Accuracy', 'Recall', 'Precision', 'F-score'])
				writer_Acc.writerow([epoch, true/count, val_recall, val_precision, val_f_score])
			else:
				dataCSVAcc = open('ValidationMetrics.csv', 'a')
				writer_Acc.writerow([epoch, true/count, val_recall, val_precision, val_f_score])
			
			#val_recall = metrics.precision_score(y_val, preds)
			#val_precision = metrics.recall_score(y_val, preds)
				
			# Then we print the results for this epoch:
			print("Epoch {} of {} took {:.3f}s".format(
				epoch + 1, num_epochs, time.time() - start_time))
			print("   Binary Classification : training loss:\t\t{:.6f}".format(train_err / train_batches))
			print("   Binary Classification : validation loss:\t\t{:.6f}".format(val_err / val_batches))
			print("   Binary Classification : validation accuracy:\t\t{:.2f} %".format(
				true/count * 100))
			print("   Binary Classification : validation recall:\t\t{:.2f} %".format(
				val_recall * 100))
			print("   Binary Classification : validation precision:\t{:.2f} %".format(
				val_precision * 100))
			print("   Binary Classification : validation f_score:\t{:.2f} %".format(
				val_f_score * 100))

		# After training, we compute and print the test metrics:
		tst_err = 0
		tst_acc = 0
		tst_rec = 0
		tst_pres = 0
		tst_batches = 0
		count = 0
		true = 0
		preds = []
		for batch in iterate_minibatches(X_tst, y_tst, instanceNames_tst, 1, shuffle=True):
			
			inputs, targets, instanceNamesTemp = batch
			err, acc, temp_preds = val_fn(inputs, targets)
			tst_err += err
			tst_acc += acc
			#print(temp_preds)
			#print(targets)
			
			if(temp_preds >= threshold and targets == 1):
				true += 1
			elif(temp_preds < threshold and targets == 0):
				true += 1
			count += 1
			
			preds.append(temp_preds[0][0].tolist())
			
			tst_batches += 1
		
		tst_recall, tst_precision = compute_recall_precision(y_tst, preds, threshold)	
		
		if(tst_precision + tst_recall == 0):
			tst_f_score = float('nan')
		else:
			tst_f_score = 2.0*tst_precision*tst_recall/ float(tst_precision+tst_recall)
			
		multiLabelClassificationResults = multilabel_fn_gather_results(preds, instanceNames_tst, instancesToMultipleLabelsMap, multiLabelClassificationResults, c, threshold)
		
		dataCSV_0 = open('predLabelsNotVehicles-Test.csv', 'wb')
		dataCSV_1 = open('predLabelsVehicles-Test.csv', 'wb')
		writer_0 = csv.writer(dataCSV_0, delimiter=',')
		writer_1 = csv.writer(dataCSV_1, delimiter=',')
		writer_0.writerow(['Predicted Label'])
		writer_1.writerow(['Predicted Label'])
		
		for label, pred_label in zip(y_tst, preds):
			if label[0] == 0:
				data = [pred_label]
				writer_0.writerow(data)		
			else:
				data = [pred_label]
				writer_1.writerow(data)		
		
		dataCSVAcc = open('TestMetrics.csv', 'wb')
		writer_Acc = csv.writer(dataCSVAcc, delimiter=',')
		writer_Acc.writerow(['Accuracy', 'Recall', 'Precision', 'F-score'])
		writer_Acc.writerow([true/count, tst_recall, tst_precision, tst_f_score])
		
		print("Final results:")
		print("   Binary Classification : test loss:\t\t\t{:.6f}".format(tst_err / tst_batches))
		print("   Binary Classification : test accuracy:\t\t{:.2f} %".format(
			true/count * 100))
		print("   Binary Classification : test recall:\t\t{:.2f} %".format(
			tst_recall * 100))
		print("   Binary Classification : test precision:\t\t{:.2f} %".format(
			tst_precision * 100))
		print("   Binary Classification : test f_score:\t{:.2f} %".format(
				tst_f_score * 100))
			
		# Optionally, you could now dump the network weights to a file like this:
		# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
		#
		# And load them again later on like this:
		# with np.load('model.npz') as f:
		#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		# lasagne.layers.set_all_param_values(network, param_values)
			
    multi_rec, multi_pres, multi_f_score = multilabel_fn_compute_metrics(multiLabelClassificationResults)
    print("   Multilabel Classification : test recall:\t\t{:.2f} %".format(multi_rec * 100))
    print("   Multilabel Classification : test precision:\t\t{:.2f} %".format(multi_pres * 100))
    print("   Multilabel Classification : test f_score:\t\t{:.2f} %".format(multi_f_score * 100))
	
	  
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
