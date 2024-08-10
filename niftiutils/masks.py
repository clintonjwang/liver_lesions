"""
Functions for manipulating binary image masks.

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/niftiutils`
"""

import copy
import cv2
import importlib
from inspect import getargvalues, currentframe
import math
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import nibabel as nib
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import niftiutils.visualization as vis
import numpy as np
import os
from joblib import Parallel, delayed
import random
import shutil
from skimage import measure
from scipy.ndimage.morphology import binary_closing, binary_opening, binary_dilation, binary_fill_holes, binary_erosion
from skimage.morphology import ball, label
import skimage.measure

importlib.reload(vis)

###########################
### Visualization
###########################

def draw_mask(mask_path, img_path=None, window=None, limit_mask_path=None):
	if img_path is not None:
		img,D = hf.nii_load(img_path)
		m = get_mask(mask_path, D, img.shape)
		if window is not None:
			img[img < window[0]] = window[0]
			img[img > window[1]] = window[1]
		img = img - img.min()

		if limit_mask_path is not None:
			M = get_mask(limit_mask_path, D, img.shape)
			crops = hf.crop_nonzero(M)[1]
			vis.draw_slices(hf.crop_nonzero(m*img, crops)[0])

		else:
			vis.draw_slices(hf.crop_nonzero(m*img)[0])
		return hf.crop_nonzero(m*img)[0]
	else:
		vis.draw_slices(hf.crop_nonzero(get_mask(mask_path))[0])

def draw_mask_contours(masks, img_path):
	if img_path is not None:
		img,D = hf.nii_load(img_path)
		m = get_mask(mask_path, D, img.shape)
		img = img - img.min()

		vis.draw_slices(hf.crop_nonzero(m*img)[0])
		return hf.crop_nonzero(m*img)[0]
	else:
		vis.draw_slices(hf.crop_nonzero(get_mask(mask_path))[0])

def create_dcm_with_mask(img_path, mask_path, save_dir, padding=None, window=None, overwrite=False):
	create_dcm_with_multimask(img_path, [mask_path], save_dir, padding, window, overwrite)
"""	importlib.reload(hf)
	img,D = hf.nii_load(img_path)
	mask = get_mask(mask_path, D, img.shape)

	if window=="ct":
		img = tr.apply_window(img)
		
	img_with_contour = np.zeros((*img.shape, 3), 'uint8')
	for sl in range(mask.shape[-1]):
		if mask[:,:,sl].sum() == 0:
			continue
		_,thresh = cv2.threshold(mask[:,:,sl],127,255,0)
		contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
		img_with_contour[:,:,sl] = cv2.drawContours(np.zeros((img.shape[0],img.shape[1],3),'uint8'), contours, -1, (255,0,0), 1)

	img -= img.min()
	img = (img/img.max()*255).astype('uint8') * (img_with_contour[...,0] == 0)
	img = np.stack([img, img, img],-1) + img_with_contour

	if padding is not None:
		mask, crops = hf.crop_nonzero(mask)

		crops = ([max(0, int(crops[0][i] - mask.shape[i]*padding)) for i in range(2)] + [max(0, int(crops[0][2] - 1))],
				[min(img.shape[i], int(crops[1][i] + mask.shape[i]*padding)) for i in range(2)] + [min(img.shape[2], int(crops[1][2] + 1))])

		img = np.stack([hf.crop_nonzero(img[...,0], crops)[0],
			hf.crop_nonzero(img[...,1], crops)[0],
			hf.crop_nonzero(img[...,2], crops)[0]], -1)

	hf.create_dicom(img, save_dir, overwrite=overwrite)
"""

def create_dcm_with_multimask(img_path, mask_paths, save_dir, padding=None, window=None, overwrite=False):
	import importlib
	importlib.reload(hf)
	img,D = hf.load_img(img_path)
	colors=[(200,0,0), (200,200,0), (0,200,0)] # if you change this, you must change img = img * (cont[...,ix] == 0)
	C=[0,0,1]

	img_with_contour = []

	if window=="ct":
		img = tr.apply_window(img)
	img -= img.min()
	img = (img/img.max()*255).astype('uint8')

	for ix, mask_path in enumerate(mask_paths):
		mask = get_mask(mask_path, D, img.shape)
		if mask.sum() == 0:
			continue
		mask = (mask/mask.max()*255).astype('uint8')
			
		img_with_contour.append(np.zeros((*img.shape, 3), 'uint8'))
		for sl in range(mask.shape[-1]):
			if mask[:,:,sl].sum() == 0:
				continue
			_,thresh = cv2.threshold(mask[:,:,sl],127,255,0)
			contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
			img_with_contour[ix][:,:,sl] = cv2.drawContours(np.zeros((img.shape[0],img.shape[1],3),'uint8'), contours, -1, colors[ix], 1)
	
	for ix,cont in enumerate(img_with_contour):
		img = img * (cont[...,C[ix]] == 0)

	img = np.stack([img, img, img],-1)
	for cont in img_with_contour:
		img = img + cont

	if padding is not None:
		mask, crops = hf.crop_nonzero(mask)

		crops = ([max(0, int(crops[0][i] - mask.shape[i]*padding)) for i in range(2)] + [max(0, int(crops[0][2] - 1))],
				[min(img.shape[i], int(crops[1][i] + mask.shape[i]*padding)) for i in range(2)] + [min(img.shape[2], int(crops[1][2] + 1))])

		img = np.stack([hf.crop_nonzero(img[...,0], crops)[0],
			hf.crop_nonzero(img[...,1], crops)[0],
			hf.crop_nonzero(img[...,2], crops)[0]], -1)

	hf.create_dicom(img, save_dir, overwrite=overwrite)

###########################
### Manipulation
###########################

def crop_vicinity(img, mask, padding=0, min_pad=0, return_crops=False, add_mask_cont=False):
	if type(img) == str:
		img,D = hf.nii_load(img)
		mask = get_mask(mask, D, img.shape)

	mask_crop, crops = hf.crop_nonzero(mask)

	padding = [max(padding, min_pad/mask_crop.shape[0]),
				max(padding, min_pad/mask_crop.shape[1]),
				np.max([.1, padding, min_pad/mask_crop.shape[2]])]

	crops = ([max(0, int(crops[0][i] - mask_crop.shape[i]*padding[i])) for i in range(3)],
			[min(img.shape[i], int(crops[1][i] + mask_crop.shape[i]*padding[i])) for i in range(3)])
	img = hf.crop_nonzero(img, crops)[0]
	mask = hf.crop_nonzero(mask, crops)[0]

	if add_mask_cont:
		if len(img.shape) == 4:
			raise ValueError('Cannot add a contour to a 4D image')
		img = np.stack([vis.create_contour_img(img[...,sl], mask[...,sl]) for sl in range(img.shape[-1])], 2)

	if return_crops:
		return img, mask
	else:
		return img

def crop_img_to_mask_vicinity(img_path, mask_path, padding=0, window=None, return_crops=False, add_mask_cont=False):	
	if type(img_path) == str:
		img,D = hf.load_img(img_path)
	else:
		img,D = img_path
	mask = get_mask(mask_path, D, img.shape)

	if window is not None:
		img = tr.apply_window(img, limits=window)

	mask_crop, crops = hf.crop_nonzero(mask)

	padding = [padding, padding, min(.1, padding)]
	crops = ([max(0, int(crops[0][i] - mask_crop.shape[i]*padding[i])) for i in range(3)],
			[min(img.shape[i], int(crops[1][i] + mask_crop.shape[i]*padding[i])) for i in range(3)])

	img, crops = hf.crop_nonzero(img, crops)
	mask = hf.crop_nonzero(mask, crops)[0]

	if add_mask_cont:
		if len(img.shape) == 4:
			raise ValueError('Cannot add a contour to a 4D image')
		img = np.stack([vis.create_contour_img(img[...,sl], mask[...,sl]) for sl in range(img.shape[-1])], 2)

	if return_crops:
		return img, crops
	else:
		return img

def create_mask_from_threshold(img, img_dims, threshold, high_mask_path=None,
			low_mask_path=None, primary_mask_path=None, binary_ops=True):
	"""Create and save a mask of orig_img based on a threshold value. Omits the threshold"""
	#img, img_dims = hf.nii_load(img_path)
	high_mask = np.zeros(img.shape)
	low_mask = np.zeros(img.shape)
	high_mask[img > threshold] = 255
	low_mask[img < threshold] = 255
	
	if binary_ops:
		high_mask = binary_opening(binary_closing(high_mask))
		low_mask = binary_opening(binary_closing(low_mask))

	if primary_mask_path is not None:
		primary_mask = get_mask(primary_mask_path, img_dims, img.shape)
		high_mask[primary_mask == 0] = 0
		low_mask[primary_mask == 0] = 0

	if high_mask_path is not None:
		high_mask = high_mask.astype('uint8')
		save_mask(high_mask, high_mask_path, vox_scales=img_dims)

	if low_mask_path is not None:
		low_mask = low_mask.astype('uint8')
		save_mask(low_mask, low_mask_path, vox_scales=img_dims)

def get_contour(img, threshold=.5):
	img = copy.deepcopy(img).astype(float)
	img = img - np.amin(img)
	img *= 255/img.max()
	threshold = round(threshold*255)
	if len(img.shape) > 2:
		img = img[:,:,img.shape[2]//2]
	img = img.astype('uint8')
	_,thresh = cv2.threshold(img,threshold,255,0)
	_,contour,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	return contour

def get_disjoint_vols(mask_path):
	mask,D = get_mask(mask_path, return_dims=True)
	V = np.product(D)

	mask_labels, num_labels = label(mask, return_num=True)
	label_sizes = [np.sum(mask_labels == label_id)*V for label_id in range(1,num_labels+1)]

	return sorted(label_sizes, reverse=True)

def split_joint_mask(mask_path, img_path, out_path):
	I,D = hf.load_img(img_path)
	mask = get_mask(mask_path, D, I.shape)[0]

	mask_labels, num_labels = label(mask, return_num=True)
	label_sizes = {label_id: np.sum(mask_labels == label_id) for label_id in range(1,num_labels+1)}

def intersection(mask1_path, mask2_path, out_path=None, img_path=None):
	mask1,D = get_mask(mask1_path, img_path=img_path, return_dims=True)
	mask2 = get_mask(mask2_path, img_path=img_path)
	M = mask1*mask2/mask2.max()

	if out_path is None:
		return M
	else:
		save_mask(M, out_path, D)

def difference(mask1_path, mask2_path, out_path=None, img_path=None):
	mask1,D = get_mask(mask1_path, img_path=img_path, return_dims=True)
	mask2 = get_mask(mask2_path, img_path=img_path)
	M = mask1*(1 - mask2/mask2.max())

	if out_path is None:
		return M
	else:
		save_mask(M, out_path, D)

def get_largest_mask(mask):
	mask_labels, num_labels = label(mask, return_num=True)
	label_sizes = [np.sum(mask_labels == label_id) for label_id in range(1,num_labels+1)]
	biggest_label = label_sizes.index(max(label_sizes))+1
	mask[mask_labels != biggest_label] = 0
	return mask

def split_mask(mask):
	mask_labels, num_labels = label(mask, return_num=True)
	label_sizes = {label_id: np.sum(mask_labels == label_id) for label_id in range(1,num_labels+1)}
	sorted_ids = sorted(label_sizes, key=label_sizes.get, reverse=True)

	M_all = []
	for m_id in sorted_ids:
		M = copy.deepcopy(mask)
		M[mask_labels != m_id] = 0
		M_all.append(M)

	return M_all

def restrict_mask_to_largest(mask_path, out_path=None, img_dims=None, img_shape=None, img_path=None):
	"""Apply the mask in mask_file to img and return the masked image.
	img_dims and img_shape can be omitted if the scaling in the mask matches that of the image"""
	if out_path is None:
		out_path = mask_path

	mask, dims = get_mask(mask_path, img_dims, img_shape, img_path, return_dims=True)
	mask = get_largest_mask(mask)
	save_mask(mask, out_path, dims)

###########################
### Subroutines
###########################

def apply_mask(orig_img, mask_file):
	"""Apply the mask in mask_file to img and return the masked image."""

	img = copy.deepcopy(orig_img)
	mask = get_mask(mask_file)

	if len(img.shape) == 4:
		for ch in img.shape[3]:
			img[...,ch][mask == 0] = 0
	else:
		img[mask == 0] = 0

	return img

def get_DICE(m1, m2):
	m1 = m1/m1.max()
	m2 = m2/m2.max()
	return .5*m1*m2/(m1+m2)

def get_volume(mask, dims):
	"""Get real volume of a binary mask with voxel dimensions dims"""
	return np.sum(mask>0) * np.prod(dims)

###########################
### OFF to IDS
###########################
	
def transpose_V(A, axes):
	B = copy.deepcopy(A)
	B[:,0] = A[:,axes[0]]
	B[:,1] = A[:,axes[1]]
	B[:,2] = A[:,axes[2]]
	return B

def off2ids_parallel(off_path, save_path=None, num_foci=None, R=[.8,.8,2.5], minVol=1, num_cores=None):
	# WARNING: may fail if not convex (if a slice has disjoint areas)
	if num_cores is None:
		num_cores = multiprocessing.cpu_count() - 1

	if save_path is None:
		save_path = off_path[:-4]

	with open(off_path, 'r') as f:
		A = np.array([x.split(' ') for x in f.readlines()[2:]])
		V = np.array([list(map(float,a)) for a in A if len(a) == 3])
		F = np.array([list(map(int,a[1:])) for a in A if len(a) == 4])
		del A

	V = sorted([v for v in split_vertices(V, F) if len(v) > 10], key=len, reverse=True)
	if num_foci is not None:
		V = V[:num_foci]

	if num_cores > 1:
		Parallel(n_jobs=num_cores)(delayed(off2ids_sub)(A, ix, save_path, num_foci,
			R, minVol) for ix, A in enumerate(V))
		return

	for ix, A in enumerate(V):
		_fill_V(A, ix, save_path, num_foci, R, minVol)
	
def off2ids_sub(A, ix, save_path, num_foci, R, minVol):
	# WARNING: may fail if not convex (if a slice has disjoint areas)
	A = A/R
	mins = A.min(0)
	maxes = A.max(0)

	G = np.zeros([math.ceil(maxes[i])-math.floor(mins[i])+1 for i in range(3)], bool)
	O = [math.floor(mins[i])*R[i] for i in range(3)]
	A -= [math.floor(x) for x in mins]
	#A = np.unique(np.round(A,1), axis=0)

	for vert in A:
		G[[slice(int(round(x)), int(round(x))+1, None) for x in vert]] = 1

	A = np.unique(np.round(A), axis=0)

	G1 = fill_2d_vertices(A,G)

	if np.sum(G1) * np.product(R) < minVol * 1000:
		return

	G = G1 * np.transpose(fill_2d_vertices(transpose_V(A, (0,2,1)), np.transpose(G, (0,2,1))), (0,2,1)) * \
		np.transpose(fill_2d_vertices(transpose_V(A, (2,1,0)), np.transpose(G, (2,1,0))), (2,1,0))

	B=np.zeros((3,3,1))
	B[:,1] = 1
	B[1,:] = 1

	if np.sum(G) * np.product(R) > 250 * 1000: #250cc
		G = binary_dilation(G, structure=B)
		G = binary_closing(G, structure=np.ones((3,3,2)))

	G = binary_fill_holes(G, structure=B)
	G = binary_closing(G)
	G = binary_erosion(G, structure=B)

	if np.sum(G1) * np.product(R) < minVol * 1000:
		return

	if num_foci == 1:
		save_mask(G, save_path, vox_scales=R, save_mesh=False, origin=O)
	else:
		save_mask(G, save_path+"_%d" % ix, vox_scales=R, save_mesh=False, origin=O)

def off2ids(off_path, save_path=None, num_foci=None, R=[.8,.8,2.5], minVol=1):
	# WARNING: may fail if not convex (if a slice has disjoint areas)
	if save_path is None:
		save_path = off_path[:-4]

	with open(off_path, 'r') as f:
		A = np.array([x.split(' ') for x in f.readlines()[2:]])
		V = np.array([list(map(float,a)) for a in A if len(a) == 3])
		F = np.array([list(map(int,a[1:])) for a in A if len(a) == 4])
		del A

	V = sorted([v for v in split_vertices(V, F) if len(v) > 10], key=len, reverse=True)
	if num_foci is not None:
		V = V[:num_foci]

	for ix, A in enumerate(V):
		A = A/R
		mins = A.min(0)
		maxes = A.max(0)

		G = np.zeros([math.ceil(maxes[i])-math.floor(mins[i])+1 for i in range(3)], bool)
		O = [math.floor(mins[i])*R[i] for i in range(3)]
		A -= [math.floor(x) for x in mins]
		#A = np.unique(np.round(A,1), axis=0)

		for vert in A:
			G[[slice(int(round(x)), int(round(x))+1, None) for x in vert]] = 1

		A = np.unique(np.round(A), axis=0)

		G1 = fill_2d_vertices(A,G)

		if np.sum(G1) * np.product(R) < minVol * 1000:
			continue

		G = G1 * np.transpose(fill_2d_vertices(transpose_V(A, (0,2,1)), np.transpose(G, (0,2,1))), (0,2,1)) * \
			np.transpose(fill_2d_vertices(transpose_V(A, (2,1,0)), np.transpose(G, (2,1,0))), (2,1,0))

		B=np.zeros((3,3,1))
		B[:,1] = 1
		B[1,:] = 1

		if np.sum(G) * np.product(R) > 250 * 1000: #250cc
			G = binary_dilation(G, structure=B)
			G = binary_closing(G, structure=np.ones((3,3,2)))

		G = binary_fill_holes(G, structure=B)
		G = binary_closing(G)
		G = binary_erosion(G, structure=B)

		if np.sum(G1) * np.product(R) < minVol * 1000:
			continue

		if num_foci == 1:
			save_mask(G, save_path, vox_scales=R, save_mesh=False, origin=O)
		else:
			save_mask(G, save_path+"_%d" % ix, vox_scales=R, save_mesh=False, origin=O)
	
def split_vertices(V, F):
	S = []
	for i in range(len(F)):
		flag = True
		for s_ix in range(len(S)):
			if len(S[s_ix].intersection(F[i])) > 0:
				S[s_ix].update(F[i])
				flag = False
				break
		if flag:
			S.append(set(F[i]))

	T = []
	cands = list(range(len(S)-1))
	for i in cands:
		flag = True
		t = S[i]
		for j in range(i+1, len(S)):
			if len(t.intersection(S[j])) > 0:
				t = t.union(S[j])
				if j in cands:
					cands.remove(j)
				flag = False

		if flag:
			T.append(S[i])
		else:
			T.append(t)
			
	return [V[np.array(list(tumor))] for tumor in T]

def clockwiseangle_and_distance(point, origin=[0,0], refvec=[0,1]):
	# Vector between point and the origin: v = p - o
	vector = [point[0]-origin[0], point[1]-origin[1]]
	# Length of vector: ||v||
	lenvector = math.hypot(vector[0], vector[1])
	# If length is zero there is no angle
	if lenvector == 0:
		return -math.pi, 0
	# Normalize vector: v/||v||
	normalized = [vector[0]/lenvector, vector[1]/lenvector]
	dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
	diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
	angle = math.atan2(diffprod, dotprod)
	# Negative angles represent counter-clockwise angles so we need to subtract them 
	# from 2*pi (360 degrees)
	if angle < 0:
		return 2*math.pi+angle, lenvector
	# I return first the angle because that's the primary sorting criterium
	# but if two vectors have the same angle then the shorter distance should come first.
	return angle, lenvector

def centroidnp(arr):
	arr = np.array(arr)
	length = arr.shape[0]
	sum_x = np.sum(arr[:, 0])
	sum_y = np.sum(arr[:, 1])
	return sum_x/length, sum_y/length

def _fill_V(ix, V2d, shape2d, out):
	if len(V2d) == 0:
		return
	V2d = sorted(V2d, key=lambda x: clockwiseangle_and_distance(x, centroidnp(V2d)))
	out = skimage.measure.grid_points_in_poly(shape2d, V2d)

def fill_2d_vertices_par(A,G, num_cores=None):
	if num_cores is None:
		num_cores = multiprocessing.cpu_count() - 1

	V2d = [[] for _ in range(G.shape[-1])]
	for ix in range(A.shape[0]):
		z = A[ix,2]
		V2d[math.floor(z)].append(A[ix,:2])
		#V2d[math.ceil(z)].append(A[ix,:2])

	M = np.zeros(G.shape)

	for ix in range(G.shape[-1]):
		if len(V2d[ix]) == 0:
			if ix == 0:
				continue
			V2d[ix] = copy.deepcopy(V2d[ix-1])

	if num_cores > 1:
		Parallel(n_jobs=num_cores)(delayed(_fill_V)(ix, V2d[ix],
			G.shape[:2], M[...,ix]) for ix in range(G.shape[-1]))
	else:
		for ix in range(G.shape[-1]):
			_fill_V(ix, V2d[ix], G.shape[:2], M[...,ix])

	return (M+G).astype(bool)

def fill_2d_vertices(A,G, num_cores=None):
	V2d = [[] for _ in range(G.shape[-1])]
	for ix in range(A.shape[0]):
		z = A[ix,2]
		V2d[math.floor(z)].append(A[ix,:2])
		#V2d[math.ceil(z)].append(A[ix,:2])

	M = np.zeros(G.shape)
	for ix in range(G.shape[-1]):
		if len(V2d[ix]) == 0:
			if ix == 0:
				continue
			V2d[ix] = copy.deepcopy(V2d[ix-1])
		V2d[ix] = sorted(V2d[ix], key=lambda x: clockwiseangle_and_distance(x, centroidnp(V2d[ix])))
		M[...,ix] = skimage.measure.grid_points_in_poly(G.shape[:2], V2d[ix])

	return (M+G).astype(bool)

###########################
### File I/O
###########################

def get_mask(mask_path, img_dims=None, img_shape=None, img_path=None, return_dims=False, overlaid=False):
	"""Apply the mask in mask_file to img and return the masked image.
	img_dims and img_shape can be omitted if the scaling in the mask matches that of the image"""

	if mask_path.find(".") != -1:
		mask_path = mask_path[:-4]
		
	if img_path is not None:
		img, img_dims = hf.load_img(img_path)
		img_shape = img.shape

	with open(mask_path + ".ics", 'r') as f:
		lns = f.readlines()
		dims = [[int(z) for z in x.split()[-3:]] for x in lns if x.startswith('layout\tsizes')][0]
		vox_dims = [[float(z) for z in x.split()[-3:]] for x in lns if x.startswith('parameter\tscale')][0]
		origin = [[float(z) for z in x.split()[-3:]] for x in lns if x.startswith('parameter\torigin')][0]

	with open(mask_path + ".ids", 'rb') as f:
		mask = f.read()
		mask = np.fromstring(mask, dtype='uint8')
		mask = np.array(mask).reshape((dims[::-1]))
		mask = np.transpose(mask, (2,1,0))

	if img_dims is not None and not np.all([abs(vox_dims[i] - img_dims[i]) < 1e-3 for i in range(3)]):
		mask = tr.rescale_img(mask, img_dims, vox_dims)

		origp = [max(int(round(origin[i]/img_dims[i])),0) for i in range(3)]
		orign = [-min(int(round(origin[i]/img_dims[i])),0) for i in range(3)]
		mask = np.pad(mask, [(origp[i],0) for i in range(3)], 'constant')
		mask = mask[orign[0]:,orign[1]:,orign[2]:]
			
		dx = [img_shape[i]-mask.shape[i] for i in range(3)]
		dxp = [max(x,0) for x in dx]
		dxn = [-min(x,0) for x in dx]

		if dxn[0] > 0:
			mask = mask[:-dxn[0],:,:]
		if dxn[1] > 0:
			mask = mask[:,:-dxn[1],:]
		if dxn[2] > 0:
			mask = mask[:,:,:-dxn[2]]

		mask = np.pad(mask, [(0,dxp[i]) for i in range(3)], 'constant')
		mask = ((mask > mask.max()/2) * 255).astype('uint8')

		vox_dims = img_dims

	if overlaid:
		mask = img*mask/mask.max()
		return mask

	if return_dims:
		return mask, vox_dims
		
	return mask.astype(bool)
	
def mask_to_mesh(mask_path, mesh_path=None):
	if mask_path.find(".") != -1:
		mask_path = mask_path[:-4]
	if mesh_path is None:
		mesh_path = mask_path + ".off"

	mask, vox_dims = get_mask(mask_path, return_dims=True)
	if mask.sum() == 0:
		print(mask_path, "is empty. No mesh will be created")
		return

	verts, faces, _, _ = measure.marching_cubes_lewiner(mask, 0, allow_degenerate=False)
	faces = measure.correct_mesh_orientation(mask, verts, faces, gradient_direction='ascent')
	for i in range(3):
		verts[:,i] = verts[:,i] * vox_dims[i]

	with open(mesh_path, "w") as f:
		f.write('OFF\n')
		f.write('%d %d 0\n' % (len(verts), len(faces)))
		for v in verts:
			f.write('%.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
		for tri in faces:
			f.write('3 %d %d %d\n' % (tri[0], tri[1], tri[2]))

def save_mask(orig_mask, mask_path, vox_scales=None, template_mask_fn=None, save_mesh=True, origin=(0,0,0)):
	"""Assumes mask is an np.ndarray. Either vox_scales or template_mask_fn must be specified"""

	if mask_path.find(".") != -1:
		mask_path = mask_path[:-4]

	mask = copy.deepcopy(orig_mask)

	if template_mask_fn is not None:
		if not template_mask_fn.endswith('.ics'):
			template_mask_fn = template_mask_fn[:template_mask_fn.find('.')] + ".ics"
		shutil.copy(template_mask_fn, mask_path + ".ics")
	else:
		with open(mask_path + ".ics", 'w') as f:
			f.writelines(["\n", "ics_version\t1.0\n", "filename\t%s\n" % (mask_path+".ids"),
				 "layout\tparameters\t4\n", "layout\torder\tbits	x	y	z\n",
				 "layout	sizes	8	%d	%d	%d\n" % mask.shape, "layout	coordinates	video\n",
				 "layout	significant_bits	8\n", "representation	format	integer\n",
				 "representation	sign	unsigned\n", "representation	SCIL_TYPE	g3d\n",
				 "parameter	origin	0	%0.3f	%0.3f	%0.3f\n" % tuple(origin),
				  "parameter	scale	1	%0.3f	%0.3f	%0.3f\n" % tuple(vox_scales),
				 "parameter	axisX	0	1.000	0.000	0.000\n",
				 "parameter	axisY	0	0.000	1.000	0.000\n",
				 "parameter	axisZ	0	0.000	0.000	1.000\n"])

	mask[mask != 0] = 255
	mask = np.transpose(mask, (2,1,0))
	mask = np.ascontiguousarray(mask).astype('uint8')
	with open(mask_path + ".ids", 'wb') as f:
		f.write(mask)

	if save_mesh:
		mask_to_mesh(mask_path)

def mask2nii(mask_path, save_path, img_path=None, img=None, img_dims=None):
	if img_path is not None:
		img, img_dims = hf.nii_load(img_path, True, True)
	mask, dims = get_mask(mask_path, img_dims=img_dims, img_shape=img.shape, return_dims=True)
	hf.save_nii(mask, save_path, dims)

def rescale_mask(mask_file, orig_dims, dims):
	"""Apply the mask in mask_file to img and return the masked image."""
	raise ValueError("rescale_mask is not ready")
	return img

