=============
 niftiutils
=============

Python library for converting dicom files to nifti

:Author: Clinton Wang
:Organization: Yale University
:Repository: https://github.com/clintonjwang/medical-img-utils

=====================
 Using niftiutils
=====================
---------------
 Installation
---------------
.. code-block:: bash

   pip install niftiutils

---------------
 Updating
---------------
.. code-block:: bash

   pip install niftiutils --upgrade

---------------
 Usage
---------------
Python
^^^^^^^^^^^^

Load a dicom file

.. code-block:: python

   import niftiutils

   img, dims = niftiutils.dcm_load(dicom_directory)

Load a nifti file

.. code-block:: python

   import dicom2nifti

   img, dims = niftiutils.ni_load(nifti_path)

----------------
 Supported data
----------------
Most anatomical data for CT and MR should be supported.