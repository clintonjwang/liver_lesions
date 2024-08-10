from setuptools import setup, find_packages

version = '0.0.2'
long_description = """
With this package you can manipulate anatomical CT and MR data in nifti format.
There is some support for 4D data (like DTI and fMRI), but the emphasis is on multiphasic MR.
"""

setup(
	name='niftiutils',
	version=version,
	description='package for manipulating nifti files',
	long_description=long_description,
	license='MIT',
	author='Clinton Wang',
	author_email='clintonjwang@gmail.com',
	#py_modules=['niftiutils.helper_fxns', 'niftiutils.transforms'],
	packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
	url='https://github.com/clintonjwang/medical-img-utils',
	keywords='nifti medical imaging',
	#scripts=['scripts/dicom2nifti'],
	# https://pypi.python.org/pypi?%3Aaction=list_classifiers
	classifiers=[
		'Development Status :: 2 - Pre-Alpha',
		'Intended Audience :: Healthcare Industry',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Medical Science Apps.',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
		'Operating System :: MacOS :: MacOS X',
		'Operating System :: POSIX :: Linux'],
	install_requires=['six', 'future', 'nibabel', 'numpy', 'pydicom>=0.9.9', 'dicom2nifti', 'matplotlib', 'pyelastix', 'SimpleITK']
	#setup_requires=['nose', 'coverage']
)
