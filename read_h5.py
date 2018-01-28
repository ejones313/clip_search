#Modified from https://stackoverflow.com/questions/28170623/how-to-read-hdf5-files-in-python
import h5py
import numpy as np
filename = 'file.hdf5'
f = h5py.File('ImageNetShuffle2016_features.h5', 'r')

# List all groups
print("Keys: ", f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = np.array(list(f[a_group_key]))
print(data)
print(data.shape)