"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/niftiutils`
"""

import keras.layers as layers
import keras.backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

####################################
### ?
####################################

def pop_n_layers(model, n=1):
	for _ in range(n):
		model.layers.pop()
		model.outputs = [model.layers[-1].output]
		model.layers[-1].outbound_nodes = []
	return model

def rm_front_layers(model, n=1):
	raise ValueError("Not ready")

####################################
### Losses
####################################

def focal_loss(y_true, y_pred, gamma=2.0, weights=4.0):
	"""
	focal loss for multi-classification
	FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
	Notice: y_pred is probability after softmax
	gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
	d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
	Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
	Focal Loss for Dense Object Detection, 130(4), 485–491.
	https://doi.org/10.1016/j.ajodo.2005.02.022
	https://github.com/zhezh/focalloss/blob/master/focalloss.py
	"""
	epsilon = 1.e-9
	y_true = K.cast(y_true, tf.float32)
	y_pred = K.cast(y_pred, tf.float32)
	y_pred = tf.add(y_pred, epsilon)
	ce = tf.multiply(y_true, -tf.log(y_pred))
	fl = tf.multiply(ce, tf.pow(tf.subtract(1., y_pred), gamma))
	#fl = K.categorical_crossentropy(y_true, y_pred)
	return K.sum(fl)