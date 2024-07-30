import numpy as np
import tensorflow as tf
import nibabel as nib
import matplotlib.pyplot as plt


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, list_ids, labels, batch_size=5, dim=(128, 128, 70), n_channels=1, n_classes=2, shuffle=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indices of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, subid in enumerate(list_ids_temp):
            nifti_vol = nib.load(subid + '.nii.gz')
            X[i, :, :, :, 0] = np.array(nifti_vol.dataobj)
            X = X.astype('float32')
            y[i] = self.labels[subid]

        return X, y
