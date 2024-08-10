"""
Functions for manipulating nifti files and other medical image data.

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/niftiutils`
"""

import copy
import cv2
import niftiutils.helper_fxns as hf
import math
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
from numba import njit
import numpy as np
import os
from os.path import *
import glob
import random
import re
from scipy.misc import imsave
import shutil
import tempfile
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time

#########################
### Plotting
#########################

def draw_img(img_path, *args):
	"""Plots slices of a 3D or 4D image loaded from a path to a numpy file, nifti file or dicom directory."""

	if img_path.endswith(".npy"):
		return draw_slices(np.load(img_path), *args)
	elif ".nii" in img_path:
		return draw_slices(nii_load(img_path)[0], *args)
	else:
		try:
			return draw_slices(dcm_load(img_path)[0], *args)
		except:
			raise ValueError(img_path + " needs to be a numpy file, nifti file or dicom directory.")

def normalize_img(img, normalize):
	img = copy.deepcopy(img)
	if normalize is not None:
		img[img < normalize[0]] = normalize[0]
		img[img > normalize[1]] = normalize[1]
		#if img.min() > normalize[0] or img.max() < normalize[1]:
		img[0,0,...] = normalize[0]
		img[0,-1,...] = normalize[1]

	return img

def draw_slices(img, normalize=None, slice_frac=0.5, slice_num=None, width=10, save_path=None):
	"""Plots slices of a 3D or 4D image.
	If 3D, outputs slices at 1/4, 1/2 and 3/4.
	If 4D, outputs middle slice for each channel.
	Can specify an optional slice_frac or slice_num argument to output a slice at a different slice_fraction.
	Can specify the width of the image in inches.
	Assumes matplotlib is already configured to display, e.g. Jupyter notebook with %matplotlib inline"""

	if slice_frac > 1 or slice_frac < 0:
		raise ValueError(slice_frac + " is not a valid slice_fraction")

	img = normalize_img(img, normalize)

	if len(img.shape) == 4:
		if slice_num is None:
			slice_num = int(img.shape[2]*slice_frac)

		num_ch = img.shape[-1]
		for ch in range(num_ch):
			plt.subplot(100 + num_ch*10 + ch+1)
			_plot_without_axes(img[:, :, slice_num, ch], width=width)

	else:
		if slice_num is not None:
			_plot_without_axes(img[:, :, slice_num], width=width)
		elif slice_frac != 0.5:
			slice_num = int(img.shape[2]*slice_frac)
			_plot_without_axes(img[:, :, slice_num], width=width)
		else:
			plt.subplot(131)
			_plot_without_axes(img[:, :, img.shape[2]//4], width=width)
			plt.subplot(132)
			_plot_without_axes(img[:, :, img.shape[2]//2], width=width)
			plt.subplot(133)
			_plot_without_axes(img[:, :, img.shape[2]*3//4], width=width)

	plt.subplots_adjust(wspace=0, hspace=0)

	if save_path is not None:
		plt.savefig(save_path, bbox_inches='tight')

def draw_multi_slices(imgs, normalize=None, slice_frac=0.5, slice_num=None, width=4, dpi=300, save_path=None):
	"""Plots slices of a 3D or 4D image.
	If 3D, outputs slices at 1/4, 1/2 and 3/4.
	If 4D, outputs middle slice for each channel.
	Can specify an optional slice_frac or slice_num argument to output a slice at a different slice_fraction.
	Can specify the width of the image in inches.
	Assumes matplotlib is already configured to display, e.g. Jupyter notebook with %matplotlib inline"""

	if slice_frac > 1 or slice_frac < 0:
		raise ValueError(slice_frac + " is not a valid slice_fraction")

	for ix, img in enumerate(imgs):
		img = copy.deepcopy(img)

		if normalize is not None:
			img[img < normalize[0]] = normalize[0]
			img[img > normalize[1]] = normalize[1]
			if img.min() > normalize[0] or img.max() < normalize[1]:
				img[0,0,...] = normalize[0]
				img[0,-1,...] = normalize[1]

		if len(img.shape) == 4:
			if slice_num is None:
				slice_num = int(img.shape[2]*slice_frac)

			num_ch = img.shape[-1]
			for ch in range(num_ch):
				plt.subplot(100 + ix*num_ch + len(imgs)*100 + num_ch*10 + ch+1)
				_plot_without_axes(img[:, :, slice_num, ch], width=width)

		else:
			if slice_num is not None:
				_plot_without_axes(img[:, :, slice_num], width=width)
			elif slice_frac != 0.5:
				slice_num = int(img.shape[2]*slice_frac)
				_plot_without_axes(img[:, :, slice_num], width=width)
			else:
				plt.subplot(131 + ix*3 + len(imgs)*100)
				_plot_without_axes(img[:, :, img.shape[2]//4], width=width)
				plt.subplot(132 + ix*3 + len(imgs)*100)
				_plot_without_axes(img[:, :, img.shape[2]//2], width=width)
				plt.subplot(133 + ix*3 + len(imgs)*100)
				_plot_without_axes(img[:, :, img.shape[2]*3//4], width=width)

	plt.subplots_adjust(wspace=0, hspace=0)

	if save_path is not None:
		plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

def draw_flipped_slices(img, voi, pad=10):
	"""Function to plot an image slice given a VOI, to test whether the z axis is flipped."""

	plt.subplot(121)
	draw_voi(img[...,0], voi, pad)
	
	plt.subplot(122)
	_plot_without_axes(img[voi['x1']-pad:voi['x2']+pad,
							voi['y2']+pad:voi['y1']-pad:-1,
							img.shape[2]-(voi['z1']+voi['z2'])//2, 0])

	plt.subplots_adjust(wspace=0, hspace=0)

def draw_voi(img, voi, pad=10, *kwargs):
	"""Draws the mid-slice of a volume of interest in an image. voi can be a tuple, list, dict or pd.Series"""

	if len(img.shape) == 4:
		if 'x1' in voi:
			draw_slices(img[voi['x1']-pad:voi['x2']+pad, voi['y1']-pad:voi['y2']+pad, voi['z1']:voi['z2'], :], *kwargs)
		else:
			x,y,z = voi
			draw_slices(img[x[0]-pad:x[1]+pad, y[0]-pad:y[1]+pad, z[0]:z[1], :], *args)

	else:
		if 'x1' in voi:
			draw_slices(img[voi['x1']-pad:voi['x2']+pad, voi['y1']-pad:voi['y2']+pad, voi['z1']:voi['z2']], *kwargs)
		else:
			x,y,z = voi
			draw_slices(img[x[0]-pad:x[1]+pad, y[0]-pad:y[1]+pad, z[0]:z[1]], *args)

def save_slice_as_img(img, filename, slice=None):
	"""Draw a slice of an image of type np array and save it to disk."""
	cnorm = matplotlib.colors.Normalize(vmin=np.amin(img), vmax=np.amax(img))
	
	if slice is None and len(img.shape)>2:
		slice=img.shape[2]//2

	w = 20
	h = int(float(img.shape[1]) / img.shape[0] * w)
	if len(img.shape)>2:
		img_slice = img[:,:,slice]
	else:
		img_slice = img

	img_slice = np.rot90(img_slice)

	fig = plt.figure(frameon=False)
	fig.set_size_inches(w,h)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(img_slice, interpolation='bilinear', norm=cnorm, cmap=plt.cm.gray, aspect='auto')

	if filename.find('.') == -1:
		filename += '.png'
	plt.savefig(filename)
	print('Slice saved as %s' % filename)
	fig.set_size_inches(w//3,h//3)
	plt.show()

def _plot_without_axes(img, cmap='gray', width=None, height=None):
	fig = plt.imshow(np.transpose(img, (1,0)), cmap=cmap)
	if width is not None:
		fig.get_figure().set_figwidth(width)
	elif height is not None:
		fig.get_figure().set_figheight(height)
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)

def display_sequence(arrs, rows, cols, save_path):
	base_ix = rows*100 + cols*10 + 1

	for ix in range(len(arrs)):
		if len(arrs[ix].shape) == 2:
			plt.subplot(base_ix+ix)
			_plot_without_axes(arrs[ix])
		else:
			plt.subplot(base_ix+ix)
			fig = plt.imshow(np.transpose(arrs[ix], (1,0,2)))
			fig.axes.get_xaxis().set_visible(False)
			fig.axes.get_yaxis().set_visible(False)

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig(save_path, dpi=150, bbox_inches='tight')

def create_contour_img(img_sl, mask_sl, colors=[(0,255,0), (255,0,0)], blend_colors=False):
	"""If blend_colors is False, the first mask in mask_sl will be overwritten
	by the second mask wherever the contour overlaps"""
	img_sl = img_sl - img_sl.min()
	img_sl = (img_sl*255/img_sl.max()).astype('uint8')
	if type(mask_sl) != list:
		mask_sl = [mask_sl]

	if len(img_sl.shape) == 3:
		sl = img_sl.shape[-1]//2
		img_sl = img_sl[...,sl]
		mask_sl = [M[...,sl] for M in mask_sl]

	if mask_sl[0].max() == 0:
		return np.transpose(np.tile(img_sl, (3,1,1)), (1,2,0))

	mask = (mask_sl[0]/mask_sl[0].max()*255).astype('uint8')
	_,thresh = cv2.threshold(mask,127,255,0)
	contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
	cont1 = cv2.drawContours(np.zeros((*img_sl.shape,3),'uint8'), contours, -1, colors[0], 1)

	img = img_sl - img_sl.min()
	img = (img/img.max()*255).astype('uint8')
	c_ix = np.argwhere(colors[0])[0,0]
	img = img * (cont1[...,c_ix] == 0)

	if len(mask_sl) == 1 or mask_sl[1].max() == 0:
		img = np.stack([img, img, img], -1)
		img += cont1
	else:
		mask = (mask_sl[1]/mask_sl[1].max()*255).astype('uint8')
		_,thresh = cv2.threshold(mask,127,255,0)
		contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
		cont2 = cv2.drawContours(np.zeros((*img_sl.shape,3),'uint8'), contours, -1, colors[1], 1)

		c_ix = np.argwhere(colors[1])[0,0]
		if not blend_colors:
			cont1[cont2[...,c_ix] != 0] = 0
		img = img * (cont2[...,c_ix] == 0)
		img = np.stack([img, img, img], -1)
		img += cont1 + cont2

	return img