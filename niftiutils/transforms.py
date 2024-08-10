"""
Functions for transforming 3D and 4D image data.

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/niftiutils`
"""

from cv2 import resize
import imutils
from inspect import currentframe
import math
import numpy as np
from niftiutils.helper_fxns import get_arg_values
import random

def rescale_img(img, target_dims, cur_dims=None):
	"""
	If cur_dims is None, rescale img to have dimensions of target_dims.
	Alternatively, rescale img to match voxel dimensions of target_dims if current voxel dimensions are cur_dims.
	"""

	if cur_dims is not None:
		vox_scale = [float(cur_dims[i]/target_dims[i]) for i in range(3)]
	else:
		vox_scale = [float(target_dims[i]/img.shape[i]) for i in range(3)]
	
	return scale3d(img, vox_scale)

def split_img(img, seg, D=[1,1,2.5], L=100, target_dims=None, filter_seg=True):
	"""Split image into cubes of length L in mm"""
	img_crops = []
	seg_crops = []

	img = rescale_img(img, [1,1,1], D)
	Rs = [range(0,img.shape[i]-L,L) for i in range(3)]
	for i in Rs[0]:
		for j in Rs[1]:
			for k in Rs[2]:
				sl = [slice(i,i+L), slice(j,j+L), slice(k,k+L)]
				if filter_seg and seg[sl].sum() == 0:
					continue
				if target_dims is None:
					img_crops.append(img[sl])
					seg_crops.append(seg[sl])
				else:
					img_crops.append(rescale_img(img[sl], target_dims))
					seg_crops.append(rescale_img(seg[sl], target_dims))

	img_crops = np.stack(img_crops, 0)
	seg_crops = np.stack(seg_crops, 0)

	return img_crops, seg_crops

def apply_window(img, wl=60, ww=400):
	"""wl is window level, ww is window width"""
	return np.clip(img, wl-ww/2, wl+ww/2)

def normalize_intensity(img, max_I=255, min_I=None, frac=1.):
	"""Normalize intensities of a 3D/4D image, scaling each channel separately (should be 4th dimension).
	frac determines the extent of normalization:
	- if frac == 1, normalizes completely to max_I and min_I.
	- if frac == 0, does not change original image.
	If min_I is None, preserves the minimum intensity of the image."""

	img = img.astype(float)

	if len(img.shape) == 4:
		for ch in range(img.shape[-1]):
			img[...,ch] = normalize_intensity(img[...,ch], *get_arg_values(currentframe())[1:])

	else:
		img_max = np.max(img)
		img_min = np.min(img)
		if img_max == img_min:
			raise ValueError("img is empty")

		target_max = max_I * frac + img_max * (1-frac)
		if min_I is None:
			target_min = img_min
		else:
			target_min = min_I * frac + img_min * (1-frac)

		img = (img - img_min) * (target_max - target_min) / (img_max - img_min) + target_min

	return img

def scale3d(img, scale, axis=-1):
	"""Stretches img by a 3D scale. img should be 3D or 4D, with the non-spatial dimension 4th.
	Does not modify img directly; returns a scaled copy."""
	img = img.astype(float)
	[scalex, scaley, scalez] = [float(x) for x in scale]

	if len(img.shape) == 4:
		scaled = np.zeros([int(round(img.shape[0] * scalex)), int(round(img.shape[1] * scaley)),
			int(round(img.shape[2] * scalez)), img.shape[3]])

		for ch in range(img.shape[axis]):
			if axis == -1:
				scaled[...,ch] = scale3d(img[...,ch], scale)
			elif axis == 0:
				scaled[ch,...] = scale3d(img[ch,...], scale)
			else:
				raise ValueError('arbitrary axis not yet supported')

	elif len(img.shape) == 3:
		inter = np.zeros([round(img.shape[0] * scalex), round(img.shape[1] * scaley), img.shape[2]])
		for s in range(img.shape[2]):
			inter[...,s] = resize(img[...,s], (0,0), fx=scaley, fy=scalex)

		if scalez == 1:
			return inter
		else:
			scaled = np.zeros([inter.shape[0], inter.shape[1], round(img.shape[2] * scalez)])
			for s in range(inter.shape[0]):
				scaled[s,...] = resize(inter[s,...], (0,0), fx=scalez, fy=1)

	else:
		raise ValueError(str(img.shape) + " should be 3D or 4D")
	
	return scaled

def rotate(img, angle, axis=-1, ch_axis=-1):
	"""Rotates a 3D or 4D img in the x-y plane (in x-y-z-ch ordering).
	The dimensions of img are preserved, so the rotated image will be cut off.
	Does not modify img directly; returns a rotated copy."""

	rotated = np.zeros(img.shape)

	if len(img.shape) == 4:
		if ch_axis == -1:
			for ch in range(img.shape[-1]):
				rotated[...,ch] = rotate(img[...,ch], angle)
		elif ch_axis == 0:
			for ch in range(img.shape[0]):
				rotated[ch,...] = rotate(img[ch,...], angle)
		else:
			raise ValueError('arbitrary axis not yet supported')

	elif len(img.shape) == 3:
		for s in range(img.shape[axis]):
			if axis == -1:
				rotated[...,s] = imutils.rotate(img[...,s], angle) #can replace rotate with rotate_bound to expand image
			elif axis == 0:
				rotated[s] = imutils.rotate(img[s], angle)
			elif axis == 1:
				rotated[:,s] = imutils.rotate(img[:,s], angle)
			else:
				raise ValueError('arbitrary axis not yet supported')

	return rotated

def offset_phases(img, max_offset=2, max_z_offset=1):
	"""Offset the channels of a 4D img by up to max_offset pixels in the
	first two dimensions and by up to max_z_offset in the 3rd dimension
	(randomly uniformly distributed). The first channel is not offset."""

	xy = max_offset+1
	z = max_z_offset+1
	img = np.pad(img, [(xy, xy), (xy, xy), (z, z), (0,0)], 'edge')

	offset_imgs = [img[xy:-xy, xy:-xy, z:-z, 0]]

	for ch in range(1, img.shape[-1]):
		offsets = [random.randint(-max_offset, max_offset),
					random.randint(-max_offset, max_offset),
					random.randint(-max_z_offset, max_z_offset)]

		offset_imgs.append(img[xy+offsets[0] : -xy+offsets[0],
							xy+offsets[1] : -xy+offsets[1],
							z+offsets[2] : -z+offsets[2], ch])


	return np.stack(offset_imgs, axis=3)
