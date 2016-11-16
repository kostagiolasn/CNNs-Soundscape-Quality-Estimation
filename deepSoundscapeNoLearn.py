#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import sys
import os
import time
import random

from sklearn.utils import shuffle

import numpy as np
import theano
import theano.tensor as T

import Image
import ImageOps
from pandas import DataFrame

import matplotlib.pyplot as plt
import lasagne
from lasagne import layers
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import linear
from lasagne.nonlinearities import sigmoid
from lasagne.updates import adam
from lasagne.updates import nesterov_momentum
from lasagne.layers import get_all_params

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective

PIXELS = 227

# ############################ load_dataset #################################
# This function simply organizes the dataset for each binary classification
# problem. It returns the training, validation and testing set each time

def load_dataset(path, className):
		
	if className == 'voice_adults':
		className = 'voice_(adults)'
	if className == 'voice_children':
		className = 'voice_(children)'
		
	negativeClassString = 'not_'
	
	positiveClass = className + '/' + className
	negativeClass = className + '/' + negativeClassString + className
	
	X_N = []
	X_P = []
	y = []
	
	positivePath = path + '/' + positiveClass
	negativePath = path + '/' + negativeClass
	
	for f in os.listdir(positivePath):
		if f.endswith(".png"):
			img = Image.open(positivePath + '/' + f)
			img = np.asarray(img, dtype='float32')# / 255
			img = img.transpose(2,0,1).reshape(4,PIXELS,PIXELS)
			X_P.append(img)
			y.append(1)
	for f in os.listdir(negativePath):
		if f.endswith(".png"):
			img = Image.open(negativePath + '/' + f)
			img = np.asarray(img, dtype='float32')# / 255
			img = img.transpose(2,0,1).reshape(4,PIXELS,PIXELS)
			X_P.append(img)
			y.append(0)
	
	X = X_N + X_P
	
	X = X + X + X
	y = y + y + y
			
	X = np.array(X).astype(np.float32)
	y = np.array(y).astype(np.int32)
	
	X, y = shuffle(X, y, random_state=42)
		
	#y = y.reshape(y.shape[0],1)
	
	X -= X.mean()
	X /= X.std()
	
	#X = X.reshape(10, 3, 227, 227)
		
	return X, y

path = '/home/nikos/Desktop/NCSRDemokritos/CNNs-Soundscape-Quality-Estimation/data/complete_dataset'
classes = ['vehicles']
	
for i in classes:
	X, y = load_dataset(path, i)

layers0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

    # first stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # second stage of our convolutional layers
    #(Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    #(Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    #(Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    #(MaxPool2DLayer, {'pool_size': 2}),

    # two dense layers with dropout
    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    # the output layer
    (DenseLayer, {'num_units': 1, 'nonlinearity': sigmoid}),
]



net0 = NeuralNet(
    layers=layers0,
    max_epochs=10,

    update=adam,
    update_learning_rate=0.0002,

    objective_l2=0.0025,

    train_split=TrainSplit(eval_size=0.25),
    verbose=1,
)

#net0.fit(X, y)

net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 4, PIXELS, PIXELS),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=100,
    output_num_units=1, output_nonlinearity=sigmoid,
    
    objective_loss_function = lasagne.objectives.binary_crossentropy,

    update_learning_rate=0.03,
    update_momentum=0.9,

    max_epochs=1000,
    verbose=1,
    )
   
#print(X)
#print(y)
net2.fit(X,y)
