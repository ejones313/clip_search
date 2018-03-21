
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import json
from pprint import pprint
from matplotlib import pyplot as plt

filename = 'sub_activitynet_v1-3.c3d.hdf5'
f = h5py.File(filename, 'r')


# In[2]:


downsample_const = 1
min_frames = 5


# In[3]:


# Build phrase objects
# Structure: 
#    dict: Key = phrase. Val = [C3D features, phrase features, vid_id].

full_objects = {}

lengths = []

for k in range(1,11):
    
    filename = 'train_vec'+str(k)+'.json'
    phrase_data = json.load(open(filename))
    
    vid_ids = list(phrase_data.keys())

    for i in vid_ids:
        c3d_features = np.array(f[i]['c3d_features'])

        phrases = phrase_data[i]["sentences"]
        vid_duration = phrase_data[i]["duration"]
        timestamps = np.array(phrase_data[i]["timestamps"])
        phrase_vecs = phrase_data[i]["vectors"]

        num_phrases = len(phrases)
        num_c3d_vecs = c3d_features.shape[0]

        indices = np.rint(num_c3d_vecs*timestamps/vid_duration).astype(int)
        for j in range(0,num_phrases):
            vid_vecs = c3d_features[np.arange(indices[j,0],indices[j,1]),:]
            num_frames = vid_vecs.shape[0]
            inds = [downsample_const * l for l in range(num_frames//downsample_const)]
            if len(inds) < min_frames:
                continue
            lengths.append(len(inds))
            vid_vecs = vid_vecs[inds]
            sentence_vecs = np.array(phrase_vecs[j])
            full_objects[phrases[j]] = [vid_vecs, sentence_vecs, i, timestamps[j]]


# In[4]:


plt.hist(np.array(lengths), bins = 28)
plt.show()


# In[6]:


import math

keys = full_objects.keys()
sets = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
       {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}] 
counter = 0
for i in keys:
    if counter < 36000:
        sets[math.floor(counter/1000)][i] = full_objects[i]
    else:
        break
    counter += 1


# In[10]:


import pickle
for i in range(len(sets)):
    filename = 'data_'+str(i)+'.pkl'
    with open(filename, 'wb') as fp:
        pickle.dump(sets[i], fp)

