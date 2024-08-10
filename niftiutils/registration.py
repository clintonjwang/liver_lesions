"""
Functions for registering medical images, with implementations for BioImageSuite (http://bioimagesuite.yale.edu/), pyelastix, and SimpleITK.

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/niftiutils`
"""

import copy
import niftiutils.helper_fxns as hf
import niftiutils.masks as masks
import niftiutils.transforms as tr
import math
import numpy as np
import os
import pyelastix
import importlib
import shutil
import SimpleITK as sitk
import subprocess
import scipy
from skimage.morphology import ball

###########################
### Higher level methods
###########################

def get_mask_Tx_shape(img_mov_path, mask_mov_path, xform_path="D:\\xform.txt", mask_path=None, ball_mask_path=None, padding=10):
	""""""
	fix_tmp_path="D:\\temp_fix.nii"
	mov_tmp_path="D:\\temp_mov.nii"

	img_m, img_m_dims = hf.nii_load(img_mov_path)
	mask_m = masks.get_mask(mask_mov_path, img_m_dims, img_m.shape)

	mask_m, crops_m = hf.crop_nonzero(mask_m)
	img_m, _ = hf.crop_nonzero(img_m, crops_m)
	img_m[mask_m == 0] = 0

	R = 1.
	mask_m_iso = tr.rescale_img(mask_m, [R]*3, img_m_dims)

	if ball_mask_path is None:
		mask_f = ball(np.mean(mask_m_iso.shape)//2)
		padding += (np.max(mask_m_iso.shape) - mask_f.shape[0])//2
		mask_f_iso = np.pad(mask_f, (np.ones((3,2))*padding).astype(int), 'constant')
	
		if mask_path is not None:
			#mask_f_iso = mask_f_iso > mask_f_iso.max()/2
			masks.save_mask(mask_f_iso, mask_path, [R]*3)
	else:
		mask_f_iso = masks.get_mask(ball_mask_path)

	mask_m_iso, pads_m = hf.zeropad(mask_m_iso, mask_f_iso.shape, ignore_neg=True, return_pads=True)

	hf.save_nii(mask_f_iso.astype(float)/mask_f_iso.max()*20-10, fix_tmp_path, [R]*3)
	hf.save_nii(mask_m_iso.astype(float)/mask_m_iso.max()*20-10, mov_tmp_path, [R]*3)

	reg_aligned_masks(fix_tmp_path, mov_tmp_path, xform_path, verbose=False)

	crops_f = np.zeros((3,2), dtype=int)
	
	return xform_path, (crops_m, crops_f), pads_m

def get_mask_Tx(img_fix_path, mask_fix_path, img_mov_path, mask_mov_path, padding=0., D=1., xform_path="D:\\xform.txt"):
	fix_tmp_path="D:\\temp_fix.nii"
	mov_tmp_path="D:\\temp_mov.nii"
	if type(D) != list:
		D = [D]*3
	
	img_f, img_f_dims = hf.nii_load(img_fix_path)
	img_m, img_m_dims = hf.nii_load(img_mov_path)
	mask_f = masks.get_mask(mask_fix_path, img_f_dims, img_f.shape)
	mask_m = masks.get_mask(mask_mov_path, img_m_dims, img_m.shape)

	# process fixed mask
	tmp, crops_f = hf.crop_nonzero(mask_f)
	pad = [tmp.shape[i]*padding for i in range(2)] + [math.ceil(padding)]
	crops_f = ([max(0, int(crops_f[0][i]-pad[i])) for i in range(3)],
				[min(img_f.shape[i]-1, int(crops_f[1][i]+pad[i])) for i in range(3)])
	mask_f_crop = hf.crop_nonzero(mask_f, crops_f)[0]
	mask_f_iso = tr.rescale_img(mask_f_crop, D, img_f_dims)
	target_shape = mask_f_iso.shape

	# process moving mask
	mask_m, crops_m = hf.crop_nonzero(mask_m)
	mask_m_tmp = tr.rescale_img(mask_m, D, img_m_dims)
	mask_m_iso, pads_m = hf.zeropad(mask_m_tmp, target_shape, ignore_neg=True, return_pads=True)

	# adjust fixed mask cropping if necessary
	if mask_m_iso.shape != target_shape:
		target_shape = mask_m_iso.shape
		_, pads_f = hf.zeropad(mask_f_iso, target_shape, ignore_neg=True, return_pads=True)
		crops_f = ([max(math.floor(crops_f[0][i] - pads_f[i][0] * D[i]/img_f_dims[i]),0) for i in range(3)],
					[math.ceil(crops_f[1][i] + pads_f[i][1] * D[i]/img_f_dims[i]) for i in range(3)])
		mask_f_crop = hf.crop_nonzero(mask_f, crops_f)[0]
		mask_f_iso = tr.rescale_img(mask_f_crop, D, img_f_dims)
		target_shape = mask_f_iso.shape

		mask_m_iso, pads_m = hf.zeropad(mask_m_tmp, target_shape, ignore_neg=True, return_pads=True)

	if not (mask_f_iso.shape == target_shape and mask_m_iso.shape == target_shape):
		raise ValueError(mask_f_iso.shape, target_shape, mask_m_iso.shape)

	# perform registration
	hf.save_nii(mask_f_iso, fix_tmp_path, D)
	hf.save_nii(mask_m_iso, mov_tmp_path, D)
	reg_aligned_masks(fix_tmp_path, mov_tmp_path, xform_path, verbose=False)

	crops_f = np.transpose(np.array(crops_f), (1,0))
	crops_f[:,1] = crops_f[:,1] + 1 - crops_f[:,0]
	
	return xform_path, (crops_m, crops_f), pads_m

def transform_region(img_path, xform_path, crops, pads_m, target_vox_dims, out_path=None, D=1., intermed_shape=None, target_shape=None):
	"""To be used after reg_mask_to_shape"""
	fix_tmp_path="D:\\temp_fix.nii"
	mov_tmp_path="D:\\temp_mov.nii"
	if type(D) != list:
		D = [D]*3

	img, img_dims = hf.nii_load(img_path)
	crops_m, crops_f = crops
	crops_m = np.array(crops_m)
	crops_m[0] = [max(round(crops_m[0][i] - pads_m[i][0] * D[i]/img_dims[i]),0) for i in range(3)]
	crops_m[1] = [round(crops_m[1][i] + pads_m[i][1] * D[i]/img_dims[i]) for i in range(3)]
	
	img, _ = hf.crop_nonzero(img, crops_m)
	img = tr.rescale_img(img, D, img_dims)

	if intermed_shape is not None:
		img = hf.crop_or_pad(img, intermed_shape)
	
	hf.save_nii(img, fix_tmp_path, D)
	transform_img(fix_tmp_path, xform_path, mov_tmp_path)
	out_img, _ = hf.nii_load(mov_tmp_path)
	
	out_img = tr.rescale_img(out_img, target_vox_dims, D)
	#out_img = np.pad(out_img, crops_f, 'constant')
	
	if target_shape is not None:
		if np.sum([abs(out_img.shape[i]-target_shape[i]) for i in range(3)]) > 6:
			raise ValueError("target_shape %s is too different from the transformed shape %s" % (target_shape, out_img.shape))
		out_img = hf.crop_or_pad(out_img, target_shape)

	if out_path is not None:
		hf.save_nii(out_img, out_path, target_vox_dims)
		
	return out_img

def transform_mask(mask_path, img_path, xform_path, crops, pads_m, target_vox_dims,
				out_path=None, D=1., intermed_shape=None, target_shape=None):
	"""To be used after get_mask_Tx"""
	fix_tmp_path="D:\\temp_fix.nii"
	mov_tmp_path="D:\\temp_mov.nii"
	if type(D) != list:
		D = [D]*3

	mask, mask_dims = masks.get_mask(mask_path, img_path=img_path, return_dims=True)
	crops_m, crops_f = crops
	
	mask, _ = hf.crop_nonzero(mask, crops_m)
	mask = tr.rescale_img(mask, D, mask_dims)
	mask = np.pad(mask, pads_m, 'constant')

	if intermed_shape is not None:
		mask = hf.crop_or_pad(mask, intermed_shape)
	
	hf.save_nii(mask, fix_tmp_path, D)
	transform_img(fix_tmp_path, xform_path, mov_tmp_path)
	out_img, _ = hf.nii_load(mov_tmp_path)
	
	if out_img.sum() == 0:
		print(mask_path, "is empty in the region of registration")
		return None

	out_img = tr.rescale_img(out_img, target_vox_dims, D)
	
	if target_shape is not None:
		if np.sum([abs(out_img.shape[i]-target_shape[i]) for i in range(3)]) > 6:
			raise ValueError("target_shape %s is too different from the transformed shape %s" % (target_shape, out_img.shape))
		out_img = hf.crop_or_pad(out_img, target_shape)

	out_img = out_img > out_img.max()/2
	#xform_mask = np.pad(xform_mask, crops_f, 'constant')
	
	if out_path is not None:
		masks.save_mask(out_img, out_path, target_vox_dims)
		
	return out_img

###########################
### Crop and register DCE-MRI
###########################

def get_gaussian_mask(shape, divisor=3):
    gauss = np.zeros(shape[:2])

    for i in range(gauss.shape[0]):
        for j in range(gauss.shape[1]):
            dx = abs(i - gauss.shape[0]/2+.5)
            dy = abs(j - gauss.shape[1]/2+.5)
            gauss[i,j] = scipy.stats.norm.pdf((dx**2 + dy**2)**.5, 0, gauss.shape[0]//divisor)
    gauss = np.transpose(np.tile(gauss, (shape[-1],1,1)), (1,2,0))

    return gauss

def get_best_sl(A,V):
    num_sl = A.shape[-1]
    best_dsl = 0
    best_dI = np.mean(np.abs(A[...,2:-2]-V[...,2:-2]))
    max_shift = num_sl//5
    for d_sl in range(1,max_shift):
        dI = np.mean(np.abs(A[...,d_sl:]-V[...,:-d_sl])) #max_shift-d_sl
        if dI < best_dI:
            best_dsl = d_sl
            best_dI = dI
    for d_sl in range(1,max_shift):
        dI = np.mean(np.abs(A[...,:-d_sl]-V[...,d_sl:]))
        if dI < best_dI:
            best_dsl = -d_sl
            best_dI = dI
    return best_dsl#, max_shift

def crop_reg(art, ven, eq, reg_type="affine", num_iter=20):
	#affine / bspline / rigid
	A = np.zeros(art.shape)
	for i in range(4,10):
		A[art > np.percentile(art,i*10)] = 1+i/10
	A *= get_gaussian_mask(art.shape, 5)

	V = np.zeros(ven.shape)
	for i in range(4,10):
		V[ven > np.percentile(ven,i*10)] = 1+i/10
	V *= get_gaussian_mask(art.shape, 5)

	E = np.zeros(eq.shape)
	for i in range(4,10):
		E[eq > np.percentile(eq,i*10)] = 1+i/10
	E *= get_gaussian_mask(art.shape, 5)

	ds1 = get_best_sl(A,V) #4
	ds2 = get_best_sl(A,E) #2

	if abs(ds1) <= 1 and abs(ds2) <= 1:
		return art, ven, eq, 0

	a_s1 = s1 = max(ds1,ds2,0)
	a_s2 = s2 = min(ds1,ds2,0)
	if s2 == 0:
		s2 = art.shape[-1]
	art = art[...,s1:s2]

	s1 = max(-ds1 + a_s1, 0)
	s2 = min(-ds1 + a_s2, 0)
	if s2 == 0:
		s2 = art.shape[-1]
	ven = ven[...,s1:s2]

	s1 = max(-ds2 + a_s1, 0)
	s2 = min(-ds2 + a_s2, 0)
	if s2 == 0:
		s2 = art.shape[-1]
	eq = eq[...,s1:s2]

	ven,_ = reg_elastix(moving=ven, fixed=art, reg_type=reg_type, num_iter=num_iter)
	eq,_ = reg_elastix(moving=eq, fixed=art, reg_type=reg_type, num_iter=num_iter)
	art = art[...,2:-2]
	ven = ven[...,2:-2]
	eq = eq[...,2:-2]

	slice_shift = a_s1 + 2

	return art, ven, eq, slice_shift	

###########################
### Subroutines
###########################

def reg_bis(fixed_img_path, moving_img_path, out_transform_path="default", out_img_path="default",
	path_to_bis="C:\\yale\\bioimagesuite35\\bin\\", overwrite=True, linear=True, settings={}):
	"""BioImageSuite. Shutil required because BIS cannot output to other drives,
	and because the output image argument is broken."""

	temp_img_path = ".\\temp_out_img.nii"
	temp_xform_path = ".\\temp_out_xform"
	
	if out_transform_path == "default":
		out_transform_path = hf.add_to_filename(moving_img_path, "-xform")

	if out_img_path == "default":
		out_img_path = hf.add_to_filename(moving_img_path, "-reg")

	if (not overwrite) and os.path.exists(out_img_path):
		print(out_img_path, "already exists. Skipping registration.")
		return None
	
	if linear:
		cmd = ''.join([path_to_bis, "bis_linearintensityregister.bat -inp ", fixed_img_path,
				  " -inp2 ", moving_img_path, " -out ", temp_xform_path]).replace("\\","/")
		for k in settings:
			cmd += " --" + k + " " + str(settings[k])
	else:
		cmd = ''.join([path_to_bis, "bis_nonlinearintensityregister.bat -inp ", fixed_img_path,
				  " -inp2 ", moving_img_path, " -out ", temp_xform_path]).replace("\\","/")
		for k in settings:
			cmd += " --" + k + " " + str(settings[k])

	subprocess.run(cmd.split())

	if not os.path.exists(temp_img_path):
		raise ValueError("Error in registration")

	if out_img_path is not None:
		shutil.copy(temp_img_path, out_img_path)
		os.remove(temp_img_path)
	if out_transform_path is not None:
		shutil.copy(temp_xform_path, out_transform_path)
		os.remove(temp_xform_path)

	return out_img_path, out_transform_path

def reg_elastix(moving, fixed, reg_type="affine", num_iter=20):
	params = pyelastix.get_default_params(type=reg_type.upper())
	params.MaximumNumberOfIterations = num_iter
	try:
		reg_img = np.ascontiguousarray(moving).astype('float32')
		fixed = np.ascontiguousarray(fixed).astype('float32')
		reg_img, field = pyelastix.register(reg_img, fixed, params, verbose=0)

	except Exception as e:
		print(e)
		fshape = fixed.shape
		mshape = moving.shape
		field = [fshape[i]/mshape[i] for i in range(3)]
		reg_img = tr.scale3d(moving, field)
		
	assert reg_img.shape == fixed.shape, ("Registration failed.")

	return reg_img, field

def reg_aligned_masks(fixed_path, moving_path, out_transform_path=None, out_img_path=None, verbose=False):
	"""Uses diffeomorphic demons"""

	fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
	moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
	
	R = sitk.DiffeomorphicDemonsRegistrationFilter()
	R.SetNumberOfIterations( 100 ) #50
	#R.SetMaximumUpdateStepLength( .5 )
	R.SetStandardDeviations( .5 )

	if verbose:
		def command_iteration(filter):
			print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(), filter.GetMetric()))
		R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

	displacementField = R.Execute( fixed, moving )
	outTx = sitk.DisplacementFieldTransform( displacementField )

	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(fixed)
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetDefaultPixelValue(0)
	resampler.SetTransform(outTx)
	out_img = resampler.Execute(moving)
	if out_transform_path is not None:
		sitk.WriteTransform(outTx, out_transform_path)
	if out_img_path is not None:
		sitk.WriteImage(out_img, out_img_path)

def transform_img(img_path, transform_path, out_img_path):
	img = sitk.ReadImage(img_path, sitk.sitkFloat32)
	xform = sitk.ReadTransform(transform_path)

	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(img)
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetDefaultPixelValue(0)
	resampler.SetTransform(xform)
	out_img = resampler.Execute(img)
	sitk.WriteImage(out_img, out_img_path)

def reg_sitk(moving_path, fixed_path, out_transform_path=None, out_img_path=None, verbose=False, reg_type="demons"):
	"""Assumes fixed and moving images are the same dimensions"""

	fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
	moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

	if reg_type == "demons":
		matcher = sitk.HistogramMatchingImageFilter()
		matcher.SetNumberOfHistogramLevels(1024)
		matcher.SetNumberOfMatchPoints(7)
		matcher.ThresholdAtMeanIntensityOn()
		moving = matcher.Execute(moving,fixed)
		
		R = sitk.DemonsRegistrationFilter()
		R.SetNumberOfIterations( 50 )
		R.SetStandardDeviations( 1.0 )

		if verbose:
			def command_iteration(filter):
				print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(), filter.GetMetric()))
			R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
	
		displacementField = R.Execute( fixed, moving )
		outTx = sitk.DisplacementFieldTransform( displacementField )

	else:
		R = sitk.ImageRegistrationMethod()

		if reg_type == 'sgd-ms':
			R.SetMetricAsMeanSquares()
			R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200 )
			R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))

		elif reg_type == 'gdls':
			fixed = sitk.Normalize(fixed)
			fixed = sitk.DiscreteGaussian(fixed, 2.0)
			moving = sitk.Normalize(moving)
			moving = sitk.DiscreteGaussian(moving, 2.0)

			R.SetMetricAsJointHistogramMutualInformation()
			R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,
										  numberOfIterations=200,
										  convergenceMinimumValue=1e-5,
										  convergenceWindowSize=5)
			R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))

		elif reg_type == 'sgd-corr':
			#doesn't work
			R.SetMetricAsCorrelation()
			R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
										   minStep=1e-4,
										   numberOfIterations=500,
										   gradientMagnitudeTolerance=1e-8 )
			R.SetOptimizerScalesFromIndexShift()
			tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform())
			R.SetInitialTransform(tx)


		elif reg_type == 'sgd-mi':
			numberOfBins = 24
			samplingPercentage = 0.10

			R.SetMetricAsMattesMutualInformation(numberOfBins)
			R.SetMetricSamplingPercentage(samplingPercentage,sitk.sitkWallClock)
			R.SetMetricSamplingStrategy(R.RANDOM)
			R.SetOptimizerAsRegularStepGradientDescent(1.0,.001,200)
			R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))


		elif reg_type == 'bspline-corr':
			transformDomainMeshSize=[8]*moving.GetDimension()
			tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize )
			
			R.SetMetricAsCorrelation()

			R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
								   numberOfIterations=100,
								   maximumNumberOfCorrections=5,
								   maximumNumberOfFunctionEvaluations=1000,
								   costFunctionConvergenceFactor=1e+7)
			R.SetInitialTransform(tx, True)

		elif reg_type == 'bspline-mi':
			transformDomainMeshSize=[10]*moving.GetDimension()
			tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize )
			
			R.SetMetricAsMattesMutualInformation(50)
			R.SetOptimizerAsGradientDescentLineSearch(5.0, 100,
													  convergenceMinimumValue=1e-4,
													  convergenceWindowSize=5)
			R.SetOptimizerScalesFromPhysicalShift( )
			R.SetInitialTransform(tx)

		elif reg_type == 'disp':
			initialTx = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(fixed.GetDimension()))

			R = sitk.ImageRegistrationMethod()

			R.SetShrinkFactorsPerLevel([3,2,1])
			R.SetSmoothingSigmasPerLevel([2,1,1])

			R.SetMetricAsJointHistogramMutualInformation(20)
			R.MetricUseFixedImageGradientFilterOff()
			R.MetricUseFixedImageGradientFilterOff()


			R.SetOptimizerAsGradientDescent(learningRate=1.0,
											numberOfIterations=100,
											estimateLearningRate = R.EachIteration)
			R.SetOptimizerScalesFromPhysicalShift()

			R.SetInitialTransform(initialTx,inPlace=True)

		elif reg_type == 'exhaust':
			R = sitk.ImageRegistrationMethod()

			R.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 50)

			sample_per_axis=12
			if fixed.GetDimension() == 2:
				tx = sitk.Euler2DTransform()
				# Set the number of samples (radius) in each dimension, with a
				# default step size of 1.0
				R.SetOptimizerAsExhaustive([sample_per_axis//2,0,0])
				# Utilize the scale to set the step size for each dimension
				R.SetOptimizerScales([2.0*pi/sample_per_axis, 1.0,1.0])
			elif fixed.GetDimension() == 3:
				tx = sitk.Euler3DTransform()
				R.SetOptimizerAsExhaustive([sample_per_axis//2,sample_per_axis//2,sample_per_axis//4,0,0,0])
				R.SetOptimizerScales([2.0*pi/sample_per_axis,2.0*pi/sample_per_axis,2.0*pi/sample_per_axis,1.0,1.0,1.0])

			# Initialize the transform with a translation and the center of
			# rotation from the moments of intensity.
			tx = sitk.CenteredTransformInitializer(fixed, moving, tx)

			R.SetInitialTransform(tx)


		R.SetInterpolator(sitk.sitkLinear)

		if reg_type == 'bspline-mi':
			R.SetShrinkFactorsPerLevel([6,2,1])
			R.SetSmoothingSigmasPerLevel([6,2,1])


		if verbose:
			def command_iteration(method) :
				print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
											   method.GetMetricValue(),
											   method.GetOptimizerPosition()))
			R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

		outTx = R.Execute(fixed, moving)

		if reg_type == 'disp':
			R.SetMovingInitialTransform(outTx)
			R.SetInitialTransform(displacementTx, inPlace=True)

			R.SetMetricAsANTSNeighborhoodCorrelation(4)
			R.MetricUseFixedImageGradientFilterOff()
			R.MetricUseFixedImageGradientFilterOff()


			R.SetShrinkFactorsPerLevel([3,2,1])
			R.SetSmoothingSigmasPerLevel([2,1,1])

			R.SetOptimizerScalesFromPhysicalShift()
			R.SetOptimizerAsGradientDescent(learningRate=1,
											numberOfIterations=300,
											estimateLearningRate=R.EachIteration)

			outTx.AddTransform( R.Execute(fixed, moving) )

	if verbose:
		print("-------")
		print(outTx)
		print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
		print(" Iteration: {0}".format(R.GetOptimizerIteration()))
		print(" Metric value: {0}".format(R.GetMetricValue()))

	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(fixed)
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetDefaultPixelValue(0)
	resampler.SetTransform(outTx)
	out_img = resampler.Execute(moving)
	if out_transform_path is not None:
		sitk.WriteTransform(outTx, out_transform_path)
	if out_img_path is not None:
		sitk.WriteImage(out_img, out_img_path)
	else:
		sitk.WriteImage(out_img, "D:\\temp.nii.gz")
	return hf.nii_load("D:\\temp.nii.gz")

def transform_bis(moving_img_path, transform_path, fixed_img_path, out_img_path="default",
	path_to_bis="C:\\yale\\bioimagesuite35\\bin\\", linear=True):
	"""Transforms based on existing transform. fixed_img_path is to define final dimensions."""
	
	temp_img_path = ".\\temp_out_img.nii"
	temp_xform_path = ".\\temp_out_xform"

	if out_img_path == "default":
		out_img_path = hf.add_to_filename(moving_img_path, "-reg")
	
	if linear:
		cmd = ''.join([path_to_bis, "bis_linearintensityregister.bat -inp ", fixed_img_path,
				  " -inp2 ", moving_img_path, " -out ", temp_xform_path,
				  " -useinitial ", transform_path, " -iterations 0"]).replace("\\","/")
	else:
		cmd = ''.join([path_to_bis, "bis_nonlinearintensityregister.bat -inp ", fixed_img_path,
				  " -inp2 ", moving_img_path, " -out ", temp_xform_path,
				  " -useinitial ", transform_path, " -iterations 0"]).replace("\\","/")

	subprocess.run(cmd.split())
	shutil.copy(temp_img_path, out_img_path)
	os.remove(temp_img_path)
	os.remove(temp_xform_path)

	return out_img_path

def transform_sitk(moving_path, transform_path, target_path=None):
	"""Transforms without scaling image"""
	
	if target_path is None:
		target_path = moving_path
	
	moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
	tx = sitk.ReadTransform(transform_path)
	moving_reg = sitk.Resample(moving, tx)
	sitk.WriteImage(moving_reg, target_path)