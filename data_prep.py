import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import pickle

class Dataset():

    def __init__(self, filename = None, anchor_is_phrase = True, data = None, cuda = False):
        if filename != None:
            self.pairs_dict = pickle.load(open(filename[0], "rb"))
            for i in range(len(filename)):
                self.pairs_dict.update(pickle.load( open( filename[i], "rb" ) ))
        elif data != None:
            self.pairs_dict = dict(data)

        self.curr_index_word = 0
        self.curr_index_vid = 0
        self.cuda = cuda

    def pairs_len(self):
        return len(list(self.pairs_dict.values()))

    def len(self, anchor_is_phrase):
        return len(self.triplets_caption) if anchor_is_phrase else len(self.triplets_clips)

    def getitem(self, index, anchor_is_phrase):
        if anchor_is_phrase:
            triplet = self.triplets_caption[index]
        else:
            triplet = self.triplets_clips[index]
        return triplet

    def retrieve_embeddings(self, start, end):
        embedding_tuples = list(self.pairs_dict.values())

        # Pairs of positive examples. First is videos, second is captions.
        dataset = [[],[]]
        lengths = [[],[]]
        num_tuples = len(embedding_tuples)

        for i in range(start, end):
            item = embedding_tuples[i]
            for type in range(2):
                dataset[type].append(item[type])
                lengths[type].append(item[type].shape[0])

        return dataset, lengths, num_tuples

    def get_pairs(self, start, end):
        #First is vids, second is captions in dataset
        dataset, lengths, num_tuples = self.retrieve_embeddings(start, end)

        # Sorted indices, first is vids, second is captions
        indices = [[],[]]

        datasets, indices = self.sort_pad_sequence(2, end - start, dataset, lengths, indices, True)

        return (dataset[1], dataset[0]), (indices[1], indices[0])

    # Pads sequences with zeros to make a square tensor.
    def pad_sequences(self, batch_size, example, length, index, backup):
        # Find max length sequence given the sorted lengths and examples
        max_len = length[0]
        if backup:
            padded = np.zeros((max_len, batch_size, example[0].shape[1]))
        else:
            padded = torch.zeros(max_len, batch_size, example[0].shape[1])

        # Effectively pads sequences with zeroes
        for i in range(batch_size):
            var = example[index[i]]
            if backup:
                padded[0:length[i], i, 0:example[0].shape[1]] = var[0:length[i],:]
            else:
                padded[0:length[i], i, 0:example[0].shape[1]] = var.data[0:length[i],:]
        return padded

    # Sorts variable length inputs and packs them into packedsequences.
    def sort_pad_sequence(self, num_types, batch_size, examples, lengths, indices, backup):
        for example_type in range(num_types):
            # For pytorch, sorts the components of the data tuples by the length of the sequence (will be unsorted correctly later)
            lengths[example_type], indices[example_type] = torch.sort(torch.IntTensor(lengths[example_type]), descending=True)
            padded = self.pad_sequences(batch_size, examples[example_type], lengths[example_type], indices[example_type], backup)

            # Convert to Variables
            if backup:
                padded = torch.from_numpy(np.array(padded))
            examples[example_type] = Variable(padded.float())

            # Backup variables if necessary
            if backup:
                if example_type == 1:
                    self.words_backup = examples[1].clone()
                elif example_type == 0:
                    self.vids_backup = examples[0].clone()


            # Obnoxious pytorch thing
            if self.cuda:
                examples[example_type] = nn.utils.rnn.pack_padded_sequence(examples[example_type].cuda(), list(lengths[example_type]))
            else:
                examples[example_type] = nn.utils.rnn.pack_padded_sequence(examples[example_type], list(lengths[example_type]))

        return examples, indices
