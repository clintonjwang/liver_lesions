"""
Functions for manipulating nifti files and other medical image data.

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/niftiutils`
"""

from dicom2nifti.convert_dicom import dicom_series_to_nifti
from dicom2nifti.convert_siemens import dicom_to_nifti
import dicom2nifti.settings as settings
import dicom2nifti.common as common
import dicom2nifti.compressed_dicom as compressed_dicom

import copy
from inspect import getargvalues, currentframe
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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pickle
import time
import SimpleITK as sitk
import errno, os, stat, shutil

###########################
### DICOM I/O
###########################

def create_dicom(arr, save_dir, D=(1,1,1), overwrite=False):
	# must be a 3D numpy array
	# Apache license https://github.com/SimpleITK/SimpleITK/blob/master/Examples/DicomSeriesReadModifyWrite/DicomSeriesReadModifySeriesWrite.py
	if len(arr.shape) == 4:
		A = np.transpose(arr[...,::-1,:], (2,1,0,3)).astype('uint8')
	else:
		A = np.transpose(arr[:,:,::-1], (2,1,0))
		if A.dtype!='uint8':
			A = A.astype('int16')

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	elif len(os.listdir(save_dir)) > 0:
		if overwrite:
			shutil.rmtree(save_dir)
			time.sleep(.1)
			while os.path.exists(save_dir):
				time.sleep(.1)
			os.makedirs(save_dir)
		else:
			print(save_dir+" already exists and is not empty.")
			return

	img = sitk.GetImageFromArray(A)
	writer = sitk.ImageFileWriter()
	writer.KeepOriginalImageUIDOn()

	modification_time = time.strftime("%H%M%S")
	modification_date = time.strftime("%Y%m%d")

	for i in range(img.GetDepth()):
		image_slice = set_metadata(img[:,:,i], i, arr, modification_time, modification_date, D)
		writer.SetFileName(os.path.join(save_dir, "%d.dcm" % i))
		writer.Execute(image_slice)

def set_metadata(image_slice, i, arr, modification_time, modification_date, D=(1,1,1)):
	image_slice.SetMetaData("0020|000d", "1.2.826.0.1.3680043.2.1125")
	image_slice.SetMetaData("0020|0013", str(i))
	# Set relevant keys indicating the change, modify or remove private tags as needed
	image_slice.SetMetaData("0008|0031", modification_time)
	image_slice.SetMetaData("0008|0021", modification_date)
	image_slice.SetMetaData("0008|0008", "DERIVED\\SECONDARY")

	#image_slice.SetMetaData("0020|0052", "1.2.826.0.1.3680043.2.1125")
	#image_slice.SetMetaData("0020|0011", '1')
	#image_slice.SetMetaData("0020|0012", '%d' % (random.random()*500))
	image_slice.SetMetaData("0020|0032", '\\'.join(['0', '0', str(i*D[2])]))
	image_slice.SetMetaData("0020|0037", '\\'.join(['1', '0', '0', '0', '1', '0']))
	image_slice.SetMetaData("0018|0050", "%.2f" % D[2])
	image_slice.SetMetaData("0018|0088", "1")
	image_slice.SetMetaData("0028|0010", str(arr.shape[0]))
	image_slice.SetMetaData("0028|0011", str(arr.shape[1]))
	image_slice.SetMetaData("0028|0030", '\\'.join(['%.3f' % D[0], '%.3f' % D[1]]))
	image_slice.SetMetaData("0020|1041", str(i))
	image_slice.SetMetaData("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time)
		
	return image_slice

def split_dcm(path2series):
	all_files = [x for x in glob.glob(join(path2series, "*")) if not x.endswith(".xml")]
	bin_time_map = {}
	fn_time_map = {}
	times = []
	H = load_dcm_header(all_files[0])[0]

	for fn in all_files:
		H = load_dcm_header(fn)[0]
		fn_time_map[fn] = int(float(H.AcquisitionTime))
		times.append(int(float(H.AcquisitionTime)))

	for ix, t in enumerate(sorted(set(times))):
		bin_time_map[t] = ix
		os.makedirs(path2series+"_bin%d" % ix)

	for fn in all_files:
		os.rename(fn, join(path2series+"_bin%d" % bin_time_map[fn_time_map[fn]], basename(fn)))

def fill_dcm_missing_slices(path2series):
	"""
	Reconstruct a DICOM directory that has inconsistent slices. Assumes only one slice is missing
	"""
	Arr = [[fn, load_dcm_header(fn)[0]] for fn in glob.glob(join(path2series, "*.dcm"))]
	Arr = sorted(Arr, key=lambda x:float(x[1].ImagePositionPatient[-1]))
	H = [[h[0],float(h[1].ImagePositionPatient[-1])] for h in Arr]
	dh = min(H[-1][1]-H[-2][1], H[1][1]-H[0][1])
	if dh == 0:
		raise ValueError("slices are not separated")

	for i in range(1,len(H)):
		if H[i][1]-H[i-1][1] - dh > 1e-3:
			print(H[i][0], H[i-1][0])
			break

	if i == len(H):
		return# False

	new_pos = copy.deepcopy(Arr[i][1].ImagePositionPatient)
	new_pos[-1] = "%.3f" % (float(new_pos[-1]) - dh)

	image_reader = sitk.ImageFileReader()
	image_reader.LoadPrivateTagsOn()
	image_list = []
	image_reader.SetFileName(H[i][0])
	img_to_dup = image_reader.Execute()

	writer = sitk.ImageFileWriter()
	writer.KeepOriginalImageUIDOn()
	modification_time = time.strftime("%H%M%S")
	modification_date = time.strftime("%Y%m%d")

	img_to_dup.SetMetaData("0020|0032", "%.3f\\%.3f\\%.3f" % (new_pos[0], new_pos[1], new_pos[2]))
	img_to_dup.SetMetaData("0008|0031", modification_time)
	img_to_dup.SetMetaData("0008|0021", modification_date)
	img_to_dup.SetMetaData("0008|0008", "DERIVED\SECONDARY")
	img_to_dup.SetMetaData("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time)
	writer.SetFileName(splitext(H[i-1][0])[0] + '_5.dcm')
	writer.Execute(img_to_dup)

	#return True

def dcm_load(path2series, flip_x=True, flip_y=True, flip_z=False, save_path=None):
	"""
	Load a dcm series as a 3D array along with its dimensions.
	Returns as a tuple:
	- the normalized (0-255) image
	- the spacing between pixels in cm
	"""

	try:
		if save_path is not None:
			dicom_series_to_nifti(path2series, save_path)
			return
			
		tmp_fn = "tmp.nii.gz"
		dicom_series_to_nifti(path2series, tmp_fn)

		ret = nii_load(tmp_fn, flip_x, flip_y, flip_z)
		os.remove(tmp_fn)
	except:
		print("Calling dcm_load_special (%s)" % path2series)
		ret = dcm_load_special(path2series, flip_x, flip_y, flip_z)

	return ret

def get_dcm_header_value(txt, search_term):
	"""Gets value corresponding to a DICOM tag in a metadata xml file
	Search_term should be formatted like '<DicomAttribute tag="00100020" vr="LO" keyword="PatientID">'
	"""
	index = txt.find(search_term) + len(search_term+'\n      <Value number="1">')
	if index == -1:
		raise ValueError(search_term, "not found")
	return txt[index:index + txt[index:].find("</Value>")]

def load_dcm_header(path):
	if path.endswith('.dcm'):
		dicom_input = [compressed_dicom.read_file(path, defer_size=100, stop_before_pixels=False, force=False)]
	else:
		dicom_input = common.read_dicom_directory(path)
	return dicom_input

@njit
def bytes_to_img(ls, bytelen):
	tmp = []
	for i,x in enumerate(range(0,len(ls),bytelen)):
		tmp.append(0)
		for y in range(bytelen):
			tmp[i] += ls[x+y]*256**y

	return tmp

def dcm_load_special(path, flip_x=True, flip_y=True, flip_z=False):
	"""Retrieves DICOM files that do not match the specs needed for dicom2nifti.
	Enforces anatomically consistent slice ordering (earlier slices are superior when flip_z is False)."""

	if path.endswith('.dcm'):
		dicom_input = [compressed_dicom.read_file(path, defer_size=100, stop_before_pixels=False, force=False)]
	else:
		dicom_input = common.read_dicom_directory(path)
	
	rows = dicom_input[0][('0028', '0010')].value
	cols = dicom_input[0][('0028', '0011')].value
	ch = dicom_input[0][('0028', '0002')].value
	try:
		frames = dicom_input[0][('0028', '0008')].value
	except:
		frames = 1
	bytelen = math.ceil(dicom_input[0][('0028', '0101')].value/8)

	if len(dicom_input) == 1:
		ls = list(dicom_input[0][('7fe0', '0010')].value)
		if len(ls) != bytelen * frames * rows * cols:
			raise ValueError("Image data does not match expected dimensions.")

		img = bytes_to_img(ls, bytelen)#[sum([ls[x+y]*256**y for y in range(bytelen)]) for x in range(0,len(ls),bytelen)]
		img = np.reshape(img,(frames,rows,cols))
		img = np.transpose(img, (2,1,0))[::-1,::-1,:]
	else:
		sl_list = []
		dicom_input = sorted(dicom_input, key=lambda x: int(x[('0020','0013')].value))

		for sl in dicom_input:
			ls = list(sl[('7fe0', '0010')].value)
			img = bytes_to_img(ls, bytelen)
			sl_list.append(np.reshape(img,(rows,cols,ch)))
		img = np.array(sl_list)
		img = np.transpose(img, (2,1,0,3))[::-1,::-1,:,:]

		if img.shape[-1] == 1:
			img = img[...,0]

		#if float(dicom_input[0][('0020', '0032')].value[-1]) > float(dicom_input[-1][('0020', '0032')].value[-1]):
		#	flip_z = not flip_z
		# this does not work...patient orientation is often flipped entirely

	try:
		voxel_dims = list(dicom_input[0][('0028', '0030')].value) + [dicom_input[0][('0018', '0050')].value]
	except:
		voxel_dims = [1,1,1]

	img = img[::(-1)**flip_x,::(-1)**flip_y,::(-1)**flip_z,...]

	return img, [float(x) for x in voxel_dims]

def save_tricolor_dcm(save_path, img_paths=None, imgs=None):
	img = []

	if imgs is None:
		if len(img_paths) == 1:
			imgs = load_img(img_paths)[0]
		else:
			imgs = [load_img(path)[0] for path in img_paths]
	else:
		imgs = [imgs[...,ix] for ix in range(imgs.shape[-1])]

	for I in imgs:
		minI = np.percentile(I,25)
		I[I < minI] = minI
		#I = tr.apply_window(I, limits=[np.percentile(I,10), np.percentile(I,95)])
		I -= I.min()
		I = I*275/I.max()
		img.append(I)
	img = np.stack(img, -1)
	img[img > 255] = 255
	create_dicom(img, save_path, overwrite=True)

###########################
### NIFTI I/O
###########################

def nii_load(filename, flip_x=False, flip_y=False, flip_z=False, normalize=False, binary=False):
	"""
	Load a nifti image as a 3D array (with optional channels) along with its dimensions.
	
	Returns as a tuple:
	- the normalized (0-255) image
	- the spacing between pixels in cm
	"""
	
	img = nib.load(filename)
	img = nib.as_closest_canonical(img) # make sure it is in the correct orientation

	dims = img.header['pixdim'][1:4]
	dim_units = img.header['xyzt_units']
	
	img = np.asarray(img.dataobj).astype(dtype='float64')
	if normalize:
		img = 255 * (img / np.amax(img))
	if binary:
		img = 255 * (img / np.amax(img))
		img = img.astype('uint8')
	
	#if dim_units == 2: #or np.sum(img) * dims[0] * dims[1] * dims[2] > 10000:
	#	dims = [d/10 for d in dims]
	img = img[::(-1)**flip_x,::(-1)**flip_y,::(-1)**flip_z]

	return img, dims

def save_nii(img, dest, dims=(1,1,1), flip_x=False, flip_y=False, flip_z=False):
	"""Saves numpy array as nifti image."""

	affine = np.eye(4)
	for i in range(3):
		affine[i,i] = dims[i]

	nii = nib.Nifti1Image(img[::(-1)**flip_x,::(-1)**flip_y,::(-1)**flip_z,...], affine)
	nib.save(nii, dest)

def flip_nii(path, flips):
	I,D = nii_load(path, flip_x=flips[0], flip_y=flips[1], flip_z=flips[2])
	save_nii(I, path, D)

###########################
### Misc I/O
###########################

def read_slicer_fcsv(fcsv_path):
	with open(fcsv_path, 'r') as f:
	    coords = np.array([x.split(',')[1:4] for x in f.readlines()[3:]], float)

	extrema = [*coords.min(0), *coords.max(0)]
	#sl = [slice(int(extrema[ix]/D[ix]), int(extrema[ix+3]/D[ix])+1) for ix in range(3)]
	return extrema

def squash_filename(fn, levels=1):
	"""Flatten directory structure by moving file up one directory"""
	target_folder = dirname(fn)
	for _ in range(levels):
		target_folder = dirname(target_folder)

	return os.rename(fn, join(target_folder, basename(fn)))

def str_to_list(s):
	for char in "['] ":
		s=s.replace(char, '')
	return [x.replace(',','') for x in s.split(",")]

def load_img(path, **kwargs):
	if '.nii' in path:
		return nii_load(path, **kwargs)
	elif '.npy' in path:
		return np.load(path)
	return dcm_load(path, **kwargs)

def pickle_dump(item, out_file):
	with open(out_file, "wb") as opened_file:
		pickle.dump(item, opened_file)

def pickle_load(in_file):
	with open(in_file, "rb") as opened_file:
		return pickle.load(opened_file)

def handleRemoveReadonly(func, path, exc):
	#try:
	#    os.remove(fn)
	#except PermissionError as e:
	#    handleRemoveReadonly(os.remove, fn, e)
	#    os.remove(fn)
	if func in (os.rmdir, os.remove) and exc.errno == errno.EACCES:
		os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
		func(path)
	else:
		raise

def sort_by_series_num(arr):
	return sorted(arr, key=lambda x: int(x[x.rfind("_")+1:]))


###########################
### Miscellaneous
###########################

def crop_nonzero(arr, crops=None, pad=None):
	if crops is None:
		nz = np.argwhere(arr)
		crops1 = nz.min(axis=0)
		crops2 = nz.max(axis=0)
	elif len(crops) == 2:
		crops1, crops2 = crops
	else:
		crops1 = crops[:,0]
		crops2 = crops[:,1] + crops[:,0] - 1

	if pad is not None:
		crops1 = [max(c-pad,0) for c in crops1]
		crops2 = [c+pad for c in crops2]
		
	return arr[crops1[0]:crops2[0]+1,
			  crops1[1]:crops2[1]+1,
			  crops1[2]:crops2[2]+1, ...], (crops1, crops2)

def crop_or_pad(img, target_shape):
	img = zeropad(img, target_shape, ignore_neg=True)
	crops = [max(img.shape[i] - target_shape[i],0) for i in range(3)]
	sl = [slice(crops[i]//2, img.shape[i] - math.ceil(crops[i]/2)) for i in range(3)]

	return img[sl[0], sl[1], sl[2]]

def zeropad(arr, target_shape, ignore_neg=False, more_padding=0, return_pads=False):
	"""If ignore_neg is False, throw an exception when target_shape is smaller than arr.
	more_padding applies additional padding to all sides"""

	pads = [target_shape[i] - arr.shape[i] for i in range(len(arr.shape))]
	if ignore_neg:
		pads = [max(p,0) + more_padding*2 for p in pads]
	pads = [(pads[i]//2, (pads[i]+1)//2) for i in range(len(arr.shape))]
	if return_pads:
		return np.pad(arr, pads, 'constant'), pads
	else:
		return np.pad(arr, pads, 'constant')

def get_hist(img, bins=None, plot_fig=True):
	"""Returns histogram in array and graphical forms."""
	h = plt.hist(flatten(img, times=len(img.shape)-1), bins=bins)
	plt.title("Histogram")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	
	#mean_intensity = np.mean(diff[img > 0])
	#std_intensity = np.std(diff[img > 0])

	return h, plt.gcf()

def flatten(l, n=1):
	"""Flattens layered lists (a list of lists or deeper) by n levels"""

	for _ in range(n):
		l = [item for sublist in l for item in sublist]
	return l

def get_arg_values(frame):
	"""Get the arguments of the current function to pass into another function that needs the same arguments.
	Good for functions whose specifications are likely to change.
	Does not handle *args or **kwargs.
	Usage:
	from inspect import currentframe
	def func(x,y,z):
		#...
		func("x_value", *get_arg_values(currentframe())[1:])
		#...
	"""
	
	mainargs, _, _, values = getargvalues(frame)
	return [values[arg] for arg in mainargs]

def init_list_of_lists(dim1, dim2, value=0):
	"""Return a list of lists of given dimensions and initialized with value."""

	lol = []
	for i in range(dim1):
		lol.append([value]*dim2)
	return lol

def add_to_filename(fn, addition):
	x = fn.find(".")
	return fn[:x] + addition + fn[x:]

def str_to_lists(raw, dtype=float):
	bigstr = str(raw)
	bigstr = re.sub(r'(\d)\.?\s+(\d)\.?', r'\1,\2', bigstr)
	bigstr = re.sub(r'(\d)\.?\s+(\d)\.?', r'\1,\2', bigstr)
	bigstr = re.sub(r'\]\s*\[', r';', bigstr)
	bigstr = bigstr.replace('[', '')
	bigstr = bigstr.replace(']', '')
	#bigstr = bigstr.replace('0. ', '0, ')
	#bigstr = bigstr.replace('1. ', '1 ')
	bigstr = bigstr.replace(' ', '')
	ret = [[dtype(x) for x in sublist.split(',')] for sublist in bigstr.split(';')]

	return ret
	