import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import pickle
import random
import utils
import math
import data_prep

def validate_L2_V2(word_model, vid_model, things, indices, cuda = False):
    vid_model.eval()
    word_model.eval()

    #Fix get_dataset to return not tuples.

    words, vids = things
    word_indices, vid_indices = indices
    vid_output = vid_model(vids)
    word_output = word_model(words)

    word_output, word_lengths = nn.utils.rnn.pad_packed_sequence(word_output)
    vid_output, vid_lengths = nn.utils.rnn.pad_packed_sequence(vid_output)
    if not cuda:
        word_unscrambled = utils.unscramble(word_output, word_lengths, word_indices, len(word_indices), cuda).data.numpy()
        vid_unscrambled = utils.unscramble(vid_output, vid_lengths, vid_indices, len(vid_indices), cuda).data.numpy()
    else:
        word_unscrambled = utils.unscramble(word_output, word_lengths, word_indices, len(word_indices), cuda).data.cpu().numpy()
        vid_unscrambled = utils.unscramble(vid_output, vid_lengths, vid_indices, len(vid_indices), cuda).data.cpu().numpy()

    dist_matrix = np.zeros((word_unscrambled.shape[0], word_unscrambled.shape[0]))
    
    for i in range(word_unscrambled.shape[0]):
        dist_matrix[i,:] = np.linalg.norm(word_unscrambled[i,:] - vid_unscrambled, axis = 1, keepdims = True).T
        
    avg_prctile_pos = np.sum(dist_matrix < np.diag(dist_matrix))/(dist_matrix.shape[0]**2)
    avg_dist_diff = np.mean((np.diag(dist_matrix) - (np.sum(dist_matrix, axis = 1) - np.diag(dist_matrix))/(dist_matrix.shape[0] - 1)))

    return float(avg_prctile_pos), float(avg_dist_diff)

def validate_L2_triplet(word_model, vid_model, things, indices, dataSet, margin, cuda = False):
    vid_model.eval()
    word_model.eval()

    words, vids = things
    word_indices, video_indices = indices

    word_output = word_model(words)
    video_output = vid_model(vids)

    # Undo pack_padded_sequence
    word_output, word_lengths = nn.utils.rnn.pad_packed_sequence(word_output)
    video_output, video_lengths = nn.utils.rnn.pad_packed_sequence(video_output)

    # Unscramble output, and unpad
    word_unscrambled = utils.unscramble(word_output, word_lengths, word_indices, len(word_indices),
                                        cuda=cuda)
    video_unscrambled = utils.unscramble(video_output, video_lengths, video_indices, len(word_indices),
                                         cuda=cuda)

    # Normalize
    #word_unscrambled = torch.nn.functional.normalize(word_unscrambled, p = 2, dim = 1)
    #video_unscrambled = torch.nn.functional.normalize(video_unscrambled, p = 2, dim = 1)

    batches, idx = dataSet.mine_triplets_all((word_unscrambled, video_unscrambled),
                                                                    (word_lengths, video_lengths), -1, margin)
    batch, indices = batches[1], idx[1]

    anchor_batch =  batch[0]
    positive_batch = batch[1]
    negative_batch = batch[2]

    anchor_indices = indices[0]
    positive_indices = indices[1]
    negative_indices = indices[2]

    # compute model output and loss, putting each component of the batch into the appropriate LSTM
    anchor_output = word_model(anchor_batch)
    positive_output = vid_model(positive_batch)
    negative_output = vid_model(negative_batch)


    #Undo pack_padded_sequence
    anchor_unpacked, anchor_lengths = nn.utils.rnn.pad_packed_sequence(anchor_output)
    positive_unpacked, positive_lengths = nn.utils.rnn.pad_packed_sequence(positive_output)
    negative_unpacked, negative_lengths = nn.utils.rnn.pad_packed_sequence(negative_output)

    #Unscramble output, and unpad
    anchor_unscrambled = utils.unscramble(anchor_unpacked, anchor_lengths, anchor_indices, anchor_unpacked.shape[1], cuda = cuda)
    positive_unscrambled = utils.unscramble(positive_unpacked, positive_lengths, positive_indices, positive_unpacked.shape[1], cuda = cuda)
    negative_unscrambled = utils.unscramble(negative_unpacked, negative_lengths, negative_indices, negative_unpacked.shape[1], cuda = cuda)
    
    #Compute loss over the batch
    loss_fn = torch.nn.modules.loss.TripletMarginLoss(margin=margin)
    total_loss = anchor_unpacked.shape[1]*loss_fn(anchor_unscrambled, positive_unscrambled, negative_unscrambled)
    potential_triplets = len(word_indices)**2 - len(word_indices)
    print("Total triplets:", potential_triplets,"    Positive distance triplets (validation):", negative_unpacked.shape[1])
    return(total_loss.data[0] / potential_triplets, negative_unpacked.shape[1] / potential_triplets)

def validate_L2(word_model, vid_model, things, indices, top_perc = 20, cuda = False):
    vid_model.eval()
    word_model.eval()

    #Fix get_dataset to return not tuples.

    words, vids = things
    word_indices, vid_indices = indices
    vid_output = vid_model(vids)
    word_output = word_model(words)

    word_output, word_lengths = nn.utils.rnn.pad_packed_sequence(word_output)
    vid_output, vid_lengths = nn.utils.rnn.pad_packed_sequence(vid_output)
    if not cuda:
        word_unscrambled = utils.unscramble(word_output, word_lengths, word_indices, len(word_indices), cuda).data.numpy()
        vid_unscrambled = utils.unscramble(vid_output, vid_lengths, vid_indices, len(vid_indices), cuda).data.numpy()
    else:
        word_unscrambled = utils.unscramble(word_output, word_lengths, word_indices, len(word_indices), cuda).data.cpu().numpy()
        vid_unscrambled = utils.unscramble(vid_output, vid_lengths, vid_indices, len(vid_indices), cuda).data.cpu().numpy()

    #word_unscrambled = word_unscrambled / np.linalg.norm(word_unscrambled, axis = 1, keepdims = True)
    #vid_unscrambled = vid_unscrambled / np.linalg.norm(vid_unscrambled, axis = 1, keepdims = True)

    top_n = int(math.floor((top_perc/100)*word_unscrambled.shape[0]))
    word_cor = 0
    vid_cor = 0

    for i in range(word_unscrambled.shape[0]):
        dists = np.linalg.norm(word_unscrambled[i,:] - vid_unscrambled, axis = 1)
        sorted_dists = np.sort(dists)
        indices = np.argsort(dists)
        if i in indices[-top_n:]:
            word_cor += 1

        dists = np.linalg.norm(vid_unscrambled[i,:] - word_unscrambled, axis = 1)
        sorted_dists = np.sort(dists)
        indices = np.argsort(dists)
        if i in indices[-top_n:]:
            vid_cor += 1

    percentage_word_good = word_cor / word_unscrambled.shape[0]
    percentage_vid_good = vid_cor / word_unscrambled.shape[0]
    precentage_total_good = (percentage_vid_good + percentage_word_good) / 2

    return percentage_word_good, percentage_vid_good, precentage_total_good


def validate(word_model, vid_model, things, indices, top_perc = 20, cuda = False):
    vid_model.eval()
    word_model.eval()

    #Fix get_dataset to return not tuples.

    words, vids = things
    word_indices, vid_indices = indices
    vid_output = vid_model(vids)
    word_output = word_model(words)

    word_output, word_lengths = nn.utils.rnn.pad_packed_sequence(word_output)
    vid_output, vid_lengths = nn.utils.rnn.pad_packed_sequence(vid_output)
    if not cuda:
        word_unscrambled = utils.unscramble(word_output, word_lengths, word_indices, len(word_indices), cuda).data.numpy()
        vid_unscrambled = utils.unscramble(vid_output, vid_lengths, vid_indices, len(vid_indices), cuda).data.numpy()
    else:
        word_unscrambled = utils.unscramble(word_output, word_lengths, word_indices, len(word_indices), cuda).data.cpu().numpy()
        vid_unscrambled = utils.unscramble(vid_output, vid_lengths, vid_indices, len(vid_indices), cuda).data.cpu().numpy()

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


def cosine_similarity(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
