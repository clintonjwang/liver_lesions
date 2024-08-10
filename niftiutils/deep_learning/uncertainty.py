# https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py

import importlib
import math
import os
import random
import time
from collections import deque
from os.path import *

import numpy as np
import tensorflow as tf
from tensorflow.contrib import distributions
import keras.activations
import keras.backend as K
import keras.layers as layers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Input, Lambda, merge, Lambda, Layer
from keras.layers.normalization import BatchNormalization
from keras.models import Model, model_from_json, Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.initializers import Constant
from keras.models import Model
from keras.regularizers import l2

import niftiutils.deep_learning.common as common

def add_aleatoric_var(pred_logits, nb_cls, lr=.001, focal_loss=0):
	"""Only applies to classification models (must output logits!)"""

	inp = Input(pred_logits.input_shape[1:], name='inp')
	y_pred = pred_logits(inp)
	y_true = Input((nb_cls,), name='y_true')
	out = AleatoricLossLayer(nb_outputs=1, nb_cls=[nb_cls], focal_loss=focal_loss)([y_true, y_pred])
	train_model = Model([inp, y_true], out)
	train_model.compile(optimizer=Adam(lr), loss=None)

	o = layers.Lambda(lambda x: x[...,:-1])(y_pred)
	o = layers.Activation('softmax')(o)
	pred_model = Model(inp, o)

	return pred_model, train_model

def add_aleatoric_var_multi(pred_model, nb_outputs, nb_cls, lr=.001):
	"""Only applies to classification models (must output logits!)"""

	inp = Input(pred_model.input_shape[1:], name='inp')
	y1_pred, y2_pred = pred_model(inp)
	y1_true = Input((*pred_model.input_shape[1:-1], C.num_segs), name='y1_true')
	y2_true = Input((len(C.cls_names),), name='y2_true')
	out = AleatoricLossLayer(nb_outputs=nb_outputs, nb_cls=nb_cls, focal_loss=1.,
				weights=C.loss_weights)([y1_true, y2_true, y1_pred, y2_pred])
	train_model = Model([inp, y1_true, y2_true], out)
	train_model.compile(optimizer=Adam(lr), loss=None)
	return pred_model, train_model

class AleatoricLossLayer(Layer):
	# This only applies to multiple classification tasks.
	# Needs to be updated to accept regression losses.
	# https://github.com/yaringal/multi-task-learning-example
	def __init__(self, nb_outputs, nb_cls, focal_loss=0, weights=None, **kwargs):
		self.nb_outputs = nb_outputs
		self.clss = list(nb_cls)
		self.focal_loss = focal_loss
		if weights is None:
			weights = [np.ones(n) for n in self.clss]
		self.W = weights

		super(AleatoricLossLayer, self).__init__(**kwargs)
		
	def build(self, input_shape=None):
		# initialise log_vars
		self.log_vars = []
		for i in range(self.nb_outputs):
			self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
											  initializer=Constant(0.), trainable=True)]

		super(AleatoricLossLayer, self).build(input_shape)

	def multi_loss(self, ys_true, ys_pred):
		assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
		loss = 0
		for y_true, y_pred, log_var, nb_cls, w in zip(ys_true, ys_pred, self.log_vars, self.clss, self.W):
			precision = K.exp(-log_var[0])
			loss += precision * self.hetero_cls_loss(y_true, y_pred, nb_cls=nb_cls, weights=w) + log_var[0]
		return K.mean(loss)

	def call(self, inputs):
		ys_true = inputs[:self.nb_outputs]
		ys_pred = inputs[self.nb_outputs:]
		loss = self.multi_loss(ys_true, ys_pred)
		self.add_loss(loss, inputs=inputs)
		# We won't actually use the output.
		return inputs#K.concatenate(inputs, -1)

	def hetero_cls_loss(self, true, pred, nb_cls, weights=None, T=500):
		# Categorical cross entropy with heteroscedastic aleatoric uncertainty.
		# https://github.com/kyle-dorman/bayesian-neural-network-blogpost
		# Does not guarantee positive variance?!
		# N data points, C classes, T monte carlo simulations
		# true - true values. Shape: (N, C)
		# pred - predicted logit values and log variance. Shape: (N, C + 1)
		# returns - loss (N,)
		if weights is None:
			weights = [1]*nb_cls

		true = K.reshape(true, [-1, nb_cls])
		pred = K.reshape(pred, [-1, nb_cls+1])
		weights = K.cast(weights, tf.float32)
		pred_scale = K.sqrt(pred[:, nb_cls:]) # shape: (N,1)
		pred = pred[:, :nb_cls] # shape: (N, C)

		dist = distributions.Laplace(loc=K.zeros_like(pred_scale), scale=pred_scale)
		#std_samples = K.transpose(dist.sample(nb_cls))
		#distorted_loss = K.categorical_crossentropy(true, pred + std_samples, from_logits=True) * weights
		
		iterable = K.variable(np.ones(T))
		mc_loss = K.mean(K.map_fn(self.aleatoric_xentropy(true, pred, dist, nb_cls), iterable, name='monte_carlo'), 0)

		return K.mean(mc_loss * weights)

	def aleatoric_xentropy(self, true, pred, dist, nb_cls):
		def map_fn(i):
			std_samples = K.transpose(dist.sample(nb_cls))
			if self.focal_loss:
				distorted_loss = common.focal_loss(true, keras.activations.softmax(pred + std_samples), self.focal_loss)
				return distorted_loss
			else:
				distorted_loss = K.categorical_crossentropy(true, pred + std_samples, from_logits=True)
				return tf.cast(K.mean(distorted_loss, -1), tf.float32)
		return map_fn
