"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/niftiutils`
Author: David G Ellis (https://github.com/ellisdg/3DUnetCNN)
"""

import keras.layers as layers
import keras.backend as K
from keras.engine.topology import Layer
from keras.regularizers import l2
import numpy as np
import tensorflow as tf

####################################
### CNN building blocks
####################################

def bn_relu_etc(L, drop=None, drop_mc=False, fc_u=None,
		cv_u=None, cv_k=(3,3,2), cv_pad='same', pool=None, order=['b','r','d']):
	if fc_u is not None and 'fc' not in order:
		L = _fc(L, fc_u)
	if cv_u is not None and 'conv' not in order:
		L = _conv3d(L, cv_u, cv_k, cv_pad)

	for o in order:
		if o == 'b':
			L = layers.BatchNormalization()(L)
		elif o == 'r':
			L = layers.Activation('relu')(L)
		elif o == 'd':
			L = _dropout(L, drop, drop_mc, cv_u is not None)
		elif o == 'prelu':
			L = layers.PReLU()(L) #trainable version of leaky RELU
		elif o == 'lrelu':
			L = layers.LeakyReLU()(L)
		elif o == 'fc':
			L = _fc(L, fc_u)
		elif o == 'conv':
			L = _conv3d(L, cv_u, cv_k, cv_pad)
		elif o == 'pool':
			L = layers.MaxPooling3D(pool)(L)

	if pool is not None and 'pool' not in order:
		L = layers.MaxPooling3D(pool)(L)

	return L

def bn_relu_conv_drop(x, nb_filter, kern=3, dropout=None, w_decay=1E-4):
	"""Apply BatchNorm, Relu, 1x1Conv, dropout. For densenet"""

	x = layers.BatchNormalization(gamma_regularizer=l2(w_decay), beta_regularizer=l2(w_decay))(x)
	x = layers.Activation('relu')(x)
	x = layers.Conv3D(nb_filter, kern,
						 kernel_initializer="he_uniform",
						 padding="same",
						 use_bias=False,
						 kernel_regularizer=l2(w_decay))(x)
	if dropout:
		x = layers.Dropout(dropout)(x)

	return x

def selu_conv(x, nb_filter, kern=3, w_decay=1E-4):
	"""For densenet"""
	x = layers.Activation('selu')(x)
	x = layers.Conv3D(nb_filter, kern,
						 kernel_initializer="lecun_normal",bias_initializer='zeros',
						 padding="same",
						 kernel_regularizer=l2(w_decay))(x)
	x = layers.AlphaDropout(.1)(x)

	return x

def spatial_sparsity(input_layer, k):
	#spatial sparsity: https://arxiv.org/pdf/1409.2752.pdf
	#https://github.com/iwyoo/tf_ConvWTA/blob/master/model.py
	#y = layers.Lambda(lambda x: tf.nn.top_k(x, 2)[0][:,-1:])(input_layer)
	#y = layers.Concatenate(axis=1)([input_layer,y])
	#return layers.Lambda(lambda x: tf.where(x[:,:-1] < x[:,-1:], tf.zeros(tf.shape(x[:,:-1])), x[:,:-1]))(y)
	return layers.Lambda(lambda x: tf.where(x < tf.nn.top_k(x, k)[0][:,-1:], tf.zeros(tf.shape(x)), x))(input_layer)

def conv_block(input_layer, f, kernel_size=(3,3,2), padding='same', dropout=.1, strides=1):
	layer = layers.Conv3D(f, kernel_size, padding=padding, activation='relu', strides=strides)(input_layer)
	#layer = layers.Lambda(lambda x: K.dropout(x, level=dropout))(layer)
	#layer = layers.Dropout(dropout)(layer)
	return layers.BatchNormalization()(layer)

"""def td_conv_block(input_layer, f, kernel_size=(3,3,2), padding='same', dropout=.1):
	layer = layers.TimeDistributed(layers.Conv3D(f, kernel_size, padding=padding, activation='relu'))(input_layer)
	return layers.TimeDistributed(layers.BatchNormalization())(layer)"""

def SeparableConv3D(input_layer, f, kernel_size=3, activation=None):
	layer = layers.Permute((4,1,2,3))(input_layer)
	layer = _expand_dims(layer)
	layer = layers.TimeDistributed(layers.Conv3D(filters=1, kernel_size=kernel_size, activation=activation))(layer)
	layer = layers.Lambda(lambda x: x[...,0])(layer)
	layer = layers.Permute((2,3,4,1))(layer)
	layer = layers.Conv3D(filters=f, kernel_size=1, activation=activation)(layer)
	return layer

def miniception(input_layer, high_f, low_f, kernel_size=(3,3,2), activation='relu'):
	layer = layers.Conv3D(filters=low_f, kernel_size=kernel_size, activation=activation)(input_layer)
	layer = layers.Conv3D(filters=high_f, kernel_size=1, activation=activation)(layer)
	layer = layers.Conv3D(filters=low_f, kernel_size=kernel_size, activation=activation)(layer)
	return layer

def Inception3D(L, f=[32,64], activation=None):
	layer_11 = layers.Conv3D(filters=f[1], kernel_size=1)(L)
	layer_33 = layers.Conv3D(filters=f[0], kernel_size=1, activation=activation)(L)
	layer_33 = layers.Conv3D(filters=f[1], kernel_size=(3,3,2), padding='same')(layer_33)
	layer_55 = layers.Conv3D(filters=f[0], kernel_size=1, activation=activation)(L)
	layer_55 = layers.Conv3D(filters=f[1], kernel_size=(5,5,3), padding='same')(layer_55)
	return layers.Concatenate()([layer_11, layer_33, layer_55])
	#layer_max = layers.MaxPooling3D((3,3,3), strides=1, padding='same')(L)
	#layer_max = layers.Conv3D(filters=f[1], kernel_size=1)(layer_max)
	#return layers.Concatenate()([layer_11, layer_33, layer_55, layer_max])

####################################
### CNN architectures
####################################

def UNet(input_layer, depth=3, base_f=32, dropout=.1, deconv=False):
	current_layer = input_layer
	levels = []

	for layer_depth in range(depth):
		layer1 = conv_block(current_layer, base_f*2**layer_depth, strides=1)
		layer2 = conv_block(layer1, base_f*2**(layer_depth+1))

		if layer_depth < depth - 1:
			current_layer = layers.MaxPooling3D(2)(layer2)
			levels.append([layer1, layer2, current_layer])
		else:
			current_layer = layer2
			levels.append([layer1, layer2])

	bottom_layer = current_layer
			
	for layer_depth in range(depth-2, -1, -1):
		if deconv:
			up_convolution = layers.Conv3DTranspose(filters=current_layer._keras_shape[1],
					kernel_size=(3,3,2), strides=pool_size)(current_layer)
		else:
			up_convolution = layers.UpSampling3D(size=pool_size)(current_layer)
		concat = layers.Concatenate(axis=-1)([up_convolution, levels[layer_depth][1]])
		current_layer = conv_block(concat, levels[layer_depth][1]._keras_shape[-1]//2)
		current_layer = conv_block(current_layer, levels[layer_depth][1]._keras_shape[-1]//2)

	return bottom_layer, current_layer


####################################
### Layer wrappers
####################################

def _conv3d(in_layer, cv_u, cv_k, cv_pad):
	return layers.Conv3D(cv_u, cv_k, padding=cv_pad, activation=None,
		kernel_initializer="he_uniform", kernel_regularizer=l2(1e-4))(in_layer)

def _fc(in_layer, fc_u):
	return layers.Dense(fc_u, activation=None, kernel_initializer="he_uniform",
		kernel_regularizer=l2(1e-4))(in_layer)

def _dropout(in_layer, dropout, mc_sampling, spatial=False):
	if dropout is not None:
		if mc_sampling:
			in_layer = layers.Lambda(lambda x: K.dropout(x, level=dropout))(in_layer)
		elif spatial:
			in_layer = layers.SpatialDropout3D(dropout)(in_layer)
		else:
			in_layer = layers.Dropout(dropout)(in_layer)
	return in_layer

def _expand_dims(in_layer, axis=-1):
	return layers.Lambda(lambda x : K.expand_dims(x, axis=axis))(in_layer)
