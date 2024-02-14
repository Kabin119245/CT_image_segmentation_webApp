import os
import glob
import nibabel as nib
import numpy as np
from scipy.interpolate import interp1d

def resample_nifti(input_path, target_num_slices):
    # Load the NIfTI file
    nifti_img = nib.load(input_path)
    data = nifti_img.get_fdata()

    # Get the current number of slices
    current_num_slices = data.shape[-1]

    # Create an interpolation function for resampling
    x_original = np.linspace(0, 1, current_num_slices)
    x_resampled = np.linspace(0, 1, target_num_slices)

    
    
      
    interpolation_func = interp1d(x_original, data, kind='linear', axis=-1, fill_value="extrapolate")

    # Resample the data using the interpolation function
    resampled_data = interpolation_func(x_resampled)

    # Create a new NIfTI image with resampled data
    new_nifti_img = nib.Nifti1Image(resampled_data, nifti_img.affine)
    nib.save(new_nifti_img, input_path)