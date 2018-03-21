import torch
import torch.nn as nn
import numpy as np
import webbrowser
import gensim
import json
import string
import data_prep
import utils
from datetime import datetime
from torch.autograd import Variable
import pickle

def unpack_and_unscramble(output, indices, batch_size = 5000):
	output, lengths = nn.utils.rnn.pad_packed_sequence(output)

	# Unscramble output
	order = torch.zeros(indices.shape).long()
	for i in range(indices.size()[0]):
		order[indices[i]] = i
	video_unscrambled = utils.unscramble(output, lengths, order, batch_size, False)
	return video_unscrambled

def main():
	data_dir = '/Users/erikjones/Documents/Stanford 2017:18/Winter CS230/Full_Data/'
	demo_dir = '/Users/erikjones/Documents/Stanford 2017:18/Winter CS230/Demo/'
	vid_model = torch.load(demo_dir + 'vid_best.pt', map_location=lambda storage, loc: storage)
	for file_num in range(12):
		print("STARTING: ", file_num)
		filenames = [data_dir + 'data_' + str(i) + '.pkl' for i in range(file_num * 3, file_num * 3 + 3)]
		print("Making dataset")
		dataSet = data_prep.Dataset(filename = filenames)
		print("Getting pairs")
		datasets, indices = dataSet.get_pairs(0, 3000, store_names = True)
		print("Done with pairs")
		vid_dataset, vid_indices = datasets[1], indices[1]
		vid_ids = dataSet.vid_ids
		timestamps = dataSet.time_stamps
		print("Passing through vid model")
		vid_outputs_packed = vid_model(vid_dataset)
		vid_embeddings = unpack_and_unscramble(vid_outputs_packed, vid_indices)
		vid_embeddings = vid_embeddings.data.numpy()
		print("Saving")
		name_root = "viddata" + str(file_num) + '_'
		vid_embeddings = np.save(name_root + 'embeddings', vid_embeddings)
		output = open(name_root + 'ids.pkl', 'wb')
		pickle.dump(vid_ids, output)
		output.close()
		output2 = open(name_root + 'timestamps.pkl', 'wb')
		pickle.dump(timestamps, output2)
		output2.close()
		print("ENDING: ", file_num)

def load_pkl_file(filename):
	with open(filename, 'rb') as pf:
		content = pickle.load(pf)
		return content

def reconstruct_files():
	timestamps = load_pkl_file("viddata0_timestamps.pkl")
	vid_ids = load_pkl_file("viddata0_ids.pkl")
	embeddings = np.load("viddata0_embeddings.npy")
	for i in range(1, 12):
		root_name = "viddata" + str(i) + "_"
		new_ts = load_pkl_file(root_name + "timestamps.pkl")
		new_ids = load_pkl_file(root_name + "ids.pkl")
		new_embeddings = np.load(root_name + "embeddings.npy")
		embeddings = np.concatenate((embeddings, new_embeddings), axis = 0)
		vid_ids += new_ids
		timestamps += new_ts
	return embeddings, vid_ids, timestamps

if __name__ == '__main__':
	start = datetime.now()
	reconstruct_files()
	print("TOTAL TIME: ", str(datetime.now() - start))



