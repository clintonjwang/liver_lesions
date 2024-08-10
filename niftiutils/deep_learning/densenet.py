# https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py

import importlib
import math
import os
import random
import time
from os.path import *

import keras.backend as K
import keras.layers as layers
import numpy as np
import tensorflow as tf
from keras.models import Model
import keras.layers as layers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Concatenate
from keras.regularizers import l2
from keras.optimizers import Adam

import niftiutils.deep_learning.cnn_components as cnnc

importlib.reload(cnnc)

def denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, w_decay=1E-4):
	for i in range(nb_layers):
		if dropout_rate == 'selu':
			merge_tensor = cnnc.selu_conv(x, growth_rate, w_decay=w_decay)
		else:
			merge_tensor = cnnc.bn_relu_conv_drop(x, growth_rate, dropout=dropout_rate, w_decay=w_decay)
		x = Concatenate()([merge_tensor, x])
		nb_filter += growth_rate

	return x, nb_filter

def DenseNet(input_shape, output_dim, optimizer='adam', depth=6, nb_dense_block=3, growth_rate=16,
		nb_filter=64, dropout_rate=None, w_decay=1E-6, non_img_inputs=0, logits=False, pool_type='avg'):
	"""DCCN"""
	model_input = Input(shape=input_shape)
	if pool_type == 'avg':
		PoolLayer = layers.AveragePooling3D
	elif pool_type == 'max':
		PoolLayer = layers.MaxPooling3D

	#assert (depth - 4) % 3 == 0, "Depth must be 3N+4"
	# layers in each dense block
	#nb_layers = int((depth - 4) / 3)
	nb_layers = depth

	# Initial convolution
	x = layers.Conv3D(nb_filter, 3,
						 kernel_initializer="he_uniform",
						 padding="same",
						 use_bias=False,
						 kernel_regularizer=l2(w_decay))(model_input)

	# Add dense blocks
	for block_idx in range(nb_dense_block - 1):
		if block_idx == 0:
			x, nb_filter = denseblock(x, 2, nb_filter, growth_rate, 
							dropout_rate=dropout_rate, w_decay=w_decay)
		else:
			x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate, 
							dropout_rate=dropout_rate, w_decay=w_decay)
		# add transition
		if dropout_rate=='selu':
			x = cnnc.selu_conv(x, nb_filter, kern=1, w_decay=w_decay)
		else:
			x = cnnc.bn_relu_conv_drop(x, nb_filter, kern=1, dropout=dropout_rate, w_decay=w_decay)
		if block_idx == 1:
			x = PoolLayer((2,2,1))(x)
		else:
			x = PoolLayer((2,2,2))(x)

	# The last denseblock does not have a transition
	x, nb_filter = denseblock(x, nb_layers,
							nb_filter, growth_rate, 
							dropout_rate=dropout_rate,
							w_decay=w_decay)

	#x = layers.BatchNormalization(gamma_regularizer=l2(w_decay),
	#											 beta_regularizer=l2(w_decay))(x)
	x = layers.Activation('relu')(x)
	x = PoolLayer((2,2,2))(x)
	x = layers.GlobalAveragePooling3D()(x)

	if non_img_inputs > 0:
		second_input = Input(shape=(non_img_inputs,))
		y = cnnc._expand_dims(second_input)
		y = layers.LocallyConnected1D(1, 1, bias_regularizer=l2(.01), activation='tanh')(y)
		y = layers.Flatten()(y)
		#y = Dense(32, activation='relu')(y)
		#y = layers.BatchNormalization()(y)
		y = Dropout(.3)(y)
		#y = ActivationLayer(activation_args)(y)
		x = Concatenate(axis=1)([x, y])

	x = Dense(output_dim, activation=None if logits else 'softmax',
			kernel_regularizer=l2(w_decay),
			bias_regularizer=l2(w_decay))(x)

	if non_img_inputs > 0:
		densenet = Model(inputs=[model_input, second_input], outputs=[x])
	else:
		densenet = Model(inputs=[model_input], outputs=[x])
	densenet.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	return densenet


"""def denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, w_decay=1E-4):
	#Build a denseblock where the output of each conv_miniblock is fed to subsequent ones

	list_feat = [x]
	for i in range(nb_layers):
		x = cnnc.bn_relu_conv_drop(x, growth_rate, dropout_rate=dropout_rate, w_decay=w_decay)
		list_feat.append(x)
		x = Concatenate()(list_feat)
		nb_filter += growth_rate

	return x, nb_filter"""
