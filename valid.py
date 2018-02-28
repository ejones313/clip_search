import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import pickle
import random


def validate(vid_model, word_model, val_data, top_n = 5):
	vid_model.eval()
	word_model.eval()

	#Fix get_dataset to return not tuples.
	things, indices = val_data.get_dataset()
	words, vids = things
	word_indices, vid_indices = indices
	vid_output = vid_model(vids)
	word_output = word_model(words)

	word_output, word_lengths = nn.utils.rnn.pad_packed_sequence(word_output)
    vid_output, vid_lengths = nn.utils.rnn.pad_packed_sequence(vid_output)
    word_unscrambled = unscramble(word_output, word_lengths, word_indices, len(word_indices))
    vid_unscrambled = unscramble(vid_output, vid_lengths, vid_indices, len(vid_indices))

    prod_matrix = torch.dot(torch.transpose(word_unscrabled), vid_unscrambled)
    print("Product matrix size: ", prod_matrix.size())

    word_sort_tenser, word_sort_ind = torch.sort(prod_matrix, axis = 1)
    vid_sort_tenser, vid_sort_ind = torch.sort(prod_matrix, axis = 0)
    top_five_vid_indices = word_sort_ind.numpy()[:,:top_n]
    top_five_word_indices = np.transpose(vid_sort_ind.numpy()[:top_n,:])

    word_loss = 0
    vid_loss = 0
    for i in range(top_five_vid_indices.shape[0]):
    	if i not in top_five_vid_indices[i]:
    		word_loss += 1
    	if i not in top_five_word_indices[i]:
    		vid_loss += 1


    percentage_word_bad = word_loss / top_five_word_indices.shape[0]
    percentage_vid_bad = vid_loss / top_five_vid_indices.shape[0]
    precentage_total_bad = (percentage_vid_bad + percentage_word_bad) / 2

    return percentage_word_bad, percentage_vid_bad, precentage_total_bad


def unscramble(output, lengths, original_indices, batch_size):
    """
    Takes the output from the model, the lengths, and original_indices, and batch size.
    Unscrambles the data, which had been sorted to make pack_padded_sequence work. 
    Returns the unsscrambled and unpadded outputs. 
    """
    final_ids = (Variable(torch.from_numpy(np.array(lengths) - 1))).view(-1,1).expand(output.size(1),output.size(2)).unsqueeze(0)
    final_outputs = output.gather(0, final_ids).squeeze()#.unsqueeze(0)

    mapping = original_indices.view(-1,1).expand(batch_size, output.size(2))
    unscrambled_outputs = final_outputs.gather(0, Variable(mapping))

    return unscrambled_outputs