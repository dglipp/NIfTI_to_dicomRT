import nibabel as nib
import numpy as np
import os

def loadNifti(path):
    '''
    Input:
        path: path string of the nifti file
    Output:
        Returns the 3D matrix of binary segmentation
        0 is background / 255 is segmented
        Returns None if any error occurs
    '''
    try:
        img = nib.load(path)
    except FileNotFoundError:
        print('[ERROR]: File not found')
    else:
        pixel_data = img.get_fdata()
        if(any(np.sort(np.unique(pixel_data)) != [0., 255.])):
            print('[ERROR]: Binary segmentation values are not 0 or 255')
            return
        if(len(pixel_data.shape) != 3):
            print('[ERROR]: Pixel map is not 3D')
            return
        return pixel_data



a = loadNifti('data/deep_GGO.nii.gz')