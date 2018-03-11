import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import pickle
import random
import utils
import math


def validate(word_model, vid_model, val_data, top_perc = 20, cuda = False):
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
    word_unscrambled = utils.unscramble(word_output, word_lengths, word_indices, len(word_indices), cuda).data.numpy()
    vid_unscrambled = utils.unscramble(vid_output, vid_lengths, vid_indices, len(vid_indices), cuda).data.numpy()

    word_norm_scale = np.linalg.inv(np.diag(np.linalg.norm(word_unscrambled, axis=1)))
    vid_norm_scale = np.linalg.inv(np.diag(np.linalg.norm(vid_unscrambled, axis=1)))
    
    prod_matrix = word_norm_scale@word_unscrambled@vid_unscrambled.T@vid_norm_scale

    top_n = int(math.floor((top_perc/100)*prod_matrix.shape[0]))

    word_sort_tenser = np.sort(prod_matrix, axis = 1)
    word_sort_ind = np.argsort(prod_matrix, axis = 1)
    vid_sort_tenser = np.sort(prod_matrix, axis = 0)
    vid_sort_ind = np.argsort(prod_matrix, axis = 0)
    top_n_vid_indices = word_sort_ind[:,-top_n:]
    top_n_word_indices = np.transpose(vid_sort_ind[-top_n:,:])


    word_cor = 0
    vid_cor = 0
    for i in range(top_n_vid_indices.shape[0]):
        if i in top_n_vid_indices[i]:
            word_cor += 1
        if i in top_n_word_indices[i]:
            vid_cor += 1        



    percentage_word_good = word_cor / top_n_word_indices.shape[0]
    percentage_vid_good = vid_cor / top_n_vid_indices.shape[0]
    precentage_total_good = (percentage_vid_good + percentage_word_good) / 2

    return percentage_word_good, percentage_vid_good, precentage_total_good
