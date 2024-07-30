import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Jayse Weaver, weaverjaysem@gmail.com
# Evaluating CNN described in https://doi.org/10.1162/imag_a_00023

# Below example processes two individual diffusion directions
# For a 4D NIfTI file, load the entire diffusion volume,
# preprocess to [128, 128, 70, # directions], and pass each
# diffusion direction volume through the loadModel.predict function
# in a loop. Record output prediction to classify each direction volume.

# load the keras model
loadModel = keras.models.load_model('models/fold1')

threshold = 0.50  # training threshold was 0.50, threshold-moving could be used to

path_list = ["example_artifact.nii.gz", "example_normal.nii.gz"]  # path to example nifti volume

for idx, filename in enumerate(path_list):
    print('evaluating: ', filename)

    # load test case
    niftiVol = nib.load(filename)

    # create appropriately sized numpy array for evaluation
    X = np.empty((1, 128, 128, 70, 1)) # data must be zero padded or cropped if not [128, 128, 70]
    X[0, :, :, :, 0] = np.array(niftiVol.dataobj)
    X.astype('float32')

    # evaluate
    pred = loadModel.predict(X)
    print(pred)

    # assign label
    if pred > threshold:
        print('Motion artifact detected')
    else:
        print('No artifact detected')
