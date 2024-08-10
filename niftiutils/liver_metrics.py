import copy
import itertools
import niftiutils.helper_fxns as hf
import niftiutils.masks as masks
import numpy as np
from scipy.ndimage.morphology import binary_closing, binary_opening, binary_dilation
import importlib

importlib.reload(masks)

def get_tumor_stats(patient_id, nii_paths, mask_paths):
	raise ValueError("out of date")
	bl_enh_mask, bl_tumor_mask, bl_voxdims, fu_enh_mask, fu_tumor_mask, fu_voxdims = get_tumor_masks(patient_id, nii_paths, mask_paths)

	stats, enh_mask, tumor_mask, voxdims = {}, bl_enh_mask, bl_tumor_mask, bl_voxdims
	stats['vRECIST'] = np.sum(tumor_mask) / 255.0 * np.product(voxdims)
	stats['qEASL'] = np.sum(enh_mask) / 255.0 * np.product(voxdims)
	stats['qEASL %'] = np.sum(enh_mask) * 100 / np.sum(tumor_mask)

	areas = [np.sum(tumor_mask[:,:,sl]) for sl in range(tumor_mask.shape[2])]
	stats['WHO'] = max(areas) / 255.0 * voxdims[0] * voxdims[1]
	max_sl = areas.index(max(areas))
	stats['RECIST'] = estimate_RECIST(tumor_mask[:,:,max_sl]) * voxdims[0]

	areas = [np.sum(enh_mask[:,:,sl]) for sl in range(enh_mask.shape[2])]
	stats['EASL'] = max(areas) / 255.0 * voxdims[0] * voxdims[1]
	max_sl = areas.index(max(areas))
	stats['mRECIST'] = estimate_RECIST(enh_mask[:,:,max_sl]) * voxdims[0]

	bl_stats = copy.deepcopy(stats)

	stats, enh_mask, tumor_mask, voxdims = {}, fu_enh_mask, fu_tumor_mask, fu_voxdims
	stats['vRECIST'] = np.sum(tumor_mask) / tumor_mask.max() * np.product(voxdims)
	stats['qEASL'] = np.sum(enh_mask) / enh_mask.max() * np.product(voxdims)
	stats['qEASL %'] = np.sum(enh_mask) * 100 / np.sum(tumor_mask)

	areas = [np.sum(tumor_mask[:,:,sl]) for sl in range(tumor_mask.shape[2])]
	stats['WHO'] = max(areas) / 255.0 * voxdims[0] * voxdims[1]
	max_sl = areas.index(max(areas))
	stats['RECIST'] = estimate_RECIST(tumor_mask[:,:,max_sl]) * voxdims[0]

	areas = [np.sum(enh_mask[:,:,sl]) for sl in range(enh_mask.shape[2])]
	stats['EASL'] = max(areas) / 255.0 * voxdims[0] * voxdims[1]
	max_sl = areas.index(max(areas))

	tumor_labels, num_labels = label(mask1, return_num=True)
	label_sizes = [np.sum(tumor_labels == label_id) for label_id in range(1,num_labels+1)]
	biggest_label = label_sizes.index(max(label_sizes))+1
	mask1[tumor_labels != biggest_label] = 0
	enh_mask = binary_closing(enh_mask)

	stats['mRECIST'] = estimate_RECIST(enh_mask[:,:,max_sl]) * voxdims[0]

	fu_stats = stats

	return {'bl': bl_stats, 'fu': fu_stats}

def get_tumor_masks(patient_id, nii_paths, mask_paths):
	raise ValueError("out of date")
	"""Assumes the tumor has not been segmented yet."""

	pat_nii_paths = nii_paths[patient_id]

	if os.path.exists(mask_paths[patient_id]["necrosis-bl"]):
		art, bl_dims = hf.nii_load(pat_nii_paths["blmri-art"])
		bl_tumor_mask = masks.get_mask(mask_paths[patient_id]["tumor-bl"], art.shape)
		bl_enh_mask = masks.get_mask(mask_paths[patient_id]["viable-tumor-bl"], art.shape)

		art, fu_dims = hf.ni_load(pat_nii_paths["fumri-art"])
		fu_tumor_mask = hf.get_mask(mask_paths[patient_id]["tumor-fu"], art.shape)
		fu_enh_mask = hf.get_mask(mask_paths[patient_id]["viable-tumor-fu"], art.shape)

	else:
		art, bl_dims = hf.ni_load(pat_nii_paths["blmri-art"])
		pre,_ = hf.ni_load(pat_nii_paths["blmri-pre"])
		liver_mask = hf.get_mask(mask_paths[patient_id]["liver-bl"], art.shape)
		bl_tumor_mask = hf.get_mask(mask_paths[patient_id]["tumor-bl"], art.shape)

		bl_enh_mask, nec_mask = segment_tumor(art, pre, liver_mask, bl_tumor_mask,
					   enh_mask_path=mask_paths[patient_id]["viable-tumor-bl"],
					   nec_mask_path=mask_paths[patient_id]["necrosis-bl"],
						 template_mask_fn=mask_paths[patient_id]["tumor-bl"])

		art, fu_dims = hf.ni_load(pat_nii_paths["fumri-art"])
		pre,_ = hf.ni_load(pat_nii_paths["fumri-pre"])
		liver_mask = hf.get_mask(mask_paths[patient_id]["liver-fu"], art.shape)
		fu_tumor_mask = hf.get_mask(mask_paths[patient_id]["tumor-fu"], art.shape)

		fu_enh_mask, fec_mask = segment_tumor(art, pre, liver_mask, fu_tumor_mask,
					   enh_mask_path=mask_paths[patient_id]["viable-tumor-fu"],
					   nec_mask_path=mask_paths[patient_id]["necrosis-fu"],
						 template_mask_fn=mask_paths[patient_id]["tumor-fu"])

	return bl_enh_mask, bl_tumor_mask, bl_dims, fu_enh_mask, fu_tumor_mask, fu_dims

def estimate_RECIST(img_slice):
	min_x = min([x for x in np.argmax(img_slice, axis=0) if x>0])
	max_x = img_slice.shape[0] - min([x for x in np.argmax(img_slice[::-1,:], axis=0) if x>0])
	min_y = min([x for x in np.argmax(img_slice, axis=1) if x>0])
	max_y = img_slice.shape[1] - min([x for x in np.argmax(img_slice[:,::-1], axis=1) if x>0])

	y_min_x = np.where(img_slice[min_x,:] > 0)[0]
	y_max_x = np.where(img_slice[max_x-1,:] > 0)[0]
	x_min_y = np.where(img_slice[:,min_y] > 0)[0]
	x_max_y = np.where(img_slice[:,max_y-1] > 0)[0]

	line=[]
	line.append(max( (min(y_min_x) - max(y_max_x))**2 , (max(y_min_x) - min(y_max_x))**2 ) + \
				(min_x - max_x)**2)

	line.append(max( (min(x_min_y) - max(x_max_y))**2 , (max(x_min_y) - min(x_max_y))**2 ) + \
				(min_y - max_y)**2)

	line.append(max( (min_y - max(y_max_x))**2 , (min_y - min(y_max_x))**2 ) + \
			max( (min(x_min_y) - max_x)**2 , (max(x_min_y) - max_x)**2 ))
		
	line.append(max( (max_y - max(y_max_x))**2 , (max_y - min(y_max_x))**2 ) + \
			max( (min(x_max_y) - max_x)**2 , (max(x_max_y) - max_x)**2 ))

	line.append(max( (min_y - max(y_min_x))**2 , (min_y - min(y_min_x))**2 ) + \
			max( (min(x_min_y) - min_x)**2 , (max(x_min_y) - min_x)**2 ))
		
	line.append(max( (max_y - max(y_min_x))**2 , (max_y - min(y_min_x))**2 ) + \
			max( (min(x_max_y) - min_x)**2 , (max(x_max_y) - min_x)**2 ))

	return max(line)**.5

###########################
### qEASLy
###########################

def segment_tumor_from_paths(art_path, pre_path, liver_mask_path, tumor_mask_path, enh_mask_path, nec_mask_path, n_bins=1000):
	art_img, vox_scales = hf.nii_load(art_path)
	pre_img, _ = hf.nii_load(pre_path)

	liver_mask = masks.get_mask(liver_mask_path, vox_scales, art_img.shape)
	tumor_mask = masks.get_mask(tumor_mask_path, vox_scales, art_img.shape)

	return segment_tumor(art_img, pre_img, liver_mask, tumor_mask, enh_mask_path, nec_mask_path, vox_scales, n_bins)

def segment_tumor(art, pre, liver_mask, tumor_mask, enh_mask_path, nec_mask_path, vox_scales, n_bins=1000, binary_ops=True):
	"""Segments the tumor based on the estimate of the parenchyma ROI for qEASL (qEASLy)."""
	
	#Find the middle of the highest histogram bin (n=1000) in the subtracted arterial image.

	try:
		art_sub = art.astype(float) - pre.astype(float)
	except ValueError:
		raise ValueError("Arterial/pre-contrast images have not been registered.")
	
	art_calc = copy.deepcopy(art_sub)
	[bin_counts, bin_edges] = np.histogram(art_calc[(liver_mask>0) & (tumor_mask==0)], bins=n_bins)
	max_bin_index = np.argmax(bin_counts)
	
	art_calc[(liver_mask==0) | (tumor_mask>0)] = np.nan
	bin_indices = np.digitize(art_calc, bin_edges) #returns right side of bin

	s = 5
	local_stds = []

	for i,j,k in itertools.product(range(art.shape[0]), range(art.shape[1]), range(art.shape[2])):
		if bin_indices[i,j,k] == max_bin_index+1:
			local_stds.append(np.nanstd(art_calc[i-s:i+s,j-s:j+s,k-s:k+s]))
	local_stds = np.array(local_stds)
	
	roi_mode = np.mean(bin_edges[max_bin_index:max_bin_index+2])
	median_std = np.median(local_stds[local_stds > 0])
	cutoff = roi_mode + 2 * median_std
	
	art_sub[tumor_mask == 0] = cutoff
	art_sub[np.isnan(art_sub)] = cutoff

	enh_mask = copy.deepcopy(tumor_mask)
	nec_mask = copy.deepcopy(tumor_mask)
	enh_mask[art_sub < cutoff] = 0
	nec_mask[art_sub > cutoff] = 0
	enh_mask = binary_opening(binary_closing(enh_mask))
	nec_mask = binary_opening(binary_closing(nec_mask))
	
	if np.sum(nec_mask) == 0:
		print("Tumor appears to have no necrosis.")
	if np.sum(enh_mask) == 0:
		raise ValueError("Tumor appears entirely necrotic.")
	else:
		masks.save_mask(enh_mask, enh_mask_path, vox_scales=vox_scales)
		masks.save_mask(nec_mask, nec_mask_path, vox_scales=vox_scales)
	
	return cutoff

def seg_tumor_from_threshold(art_path, pre_path, threshold, tumor_mask_path, enh_mask_path, nec_mask_path, binary_ops=True):
	art_img, vox_dims = hf.nii_load(art_path)
	if pre_path is not None:
		pre_img, _ = hf.nii_load(pre_path)
	try:
		if pre_path is not None:
			art_sub = art_img.astype(float) - pre_img.astype(float)
		else:
			art_sub = art_img.astype(float)
	except ValueError:
		raise ValueError("Arterial/pre-contrast images have not been registered.")
	#art_sub[tumor_mask == 0] = threshold

	masks.create_mask_from_threshold(art_sub, vox_dims, threshold, high_mask_path=enh_mask_path,
                           low_mask_path=nec_mask_path, primary_mask_path=tumor_mask_path, binary_ops=binary_ops)
