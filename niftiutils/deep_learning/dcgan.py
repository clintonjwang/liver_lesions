from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
import keras.layers as layers
import importlib
import matplotlib.pyplot as plt
import sys
from os.path import *
import numpy as np

import niftiutils.deep_learning.cnn_components as cnnc

class DCGAN():
	def __init__(self, fig_path, latent_dims, G_func, D_func):
		self.fig_path = fig_path
		self.latent_dims = latent_dims
		self.build_generator = G_func
		self.build_discriminator = D_func

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy',
			optimizer=Adam(0.0003), metrics=['accuracy'])

		# Build the generator
		self.generator = self.build_generator()
		self.generator.compile(loss='mse', optimizer=Adam(0.0008))

		z = Input((self.latent_dims,))
		img = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The discriminator takes generated images as input and determines validity
		valid = self.discriminator(img)

		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0008))


	def train(self, epochs, gen, save_interval=200):
		#generator should yield (latent vector, real image)
		for epoch in range(epochs):
			latent_vec, real_img = next(gen)
			batch_size = latent_vec.shape[0]

			if epoch % 10 == 0:
				#self.discriminator.trainable = True
				gen_imgs = self.generator.predict(latent_vec)
				# Train the discriminator (real imgs classified as ones and generated as zeros)
				d_loss_real = self.discriminator.train_on_batch(real_img, np.ones((batch_size, 1)))
				d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
				d_loss = .5*np.add(d_loss_real, d_loss_fake)
				#self.discriminator.trainable = False
			else:
				d_loss = [0, 0]

			# Train the generator (wants discriminator to mistake images as real)s
			g_loss = self.generator.train_on_batch(latent_vec, real_img)
			if np.isnan(g_loss):
				raise ValueError()
			c_loss = self.combined.train_on_batch(latent_vec, np.ones((batch_size, 1)))

			# Plot the progress
			if epoch % 50 == 0:
				print ("%d [D loss: %.3f, acc.: %d%%] [G loss: %.3f] [C loss: %.3f]" % \
						(epoch, d_loss[0], 100*d_loss[1], g_loss, c_loss))

			# If at save interval => save generated image samples
			if epoch % save_interval == 0:
				self.save_imgs(gen_imgs, epoch)

	def g_train(self, epochs, gen, save_interval=1000):
		#generator should yield (latent vector, real image)
		for epoch in range(epochs):
			latent_vec, real_img = next(gen)
			batch_size = latent_vec.shape[0]

			d_loss = [0, 0]
			c_loss = 0

			# Train the generator (wants discriminator to mistake images as real)s
			g_loss = self.generator.train_on_batch(latent_vec, real_img)
			if np.isnan(g_loss):
				raise ValueError()
			c_loss = self.combined.train_on_batch(latent_vec, np.ones((batch_size, 1)))

			# Plot the progress
			if epoch % 50 == 0:
				print ("%d [G loss: %.3f] [C loss: %.3f]" % \
						(epoch, g_loss, c_loss))


	def save_imgs(self, gen_imgs, epoch):
		r,c = 2,len(gen_imgs)//2
		fig, axs = plt.subplots(r, c)
		#fig.suptitle("DCGAN: Generated digits", fontsize=12)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt,:,:,gen_imgs.shape[3]//2], cmap='gray')
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig(join(self.fig_path, "gen_imgs_%d.png") % epoch)
		plt.close()
