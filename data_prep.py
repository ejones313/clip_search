import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import pickle
import random

class Dataset(data.Dataset):

    def __init__(self, filename = None, anchor_is_phrase = True, data = None):
        if filename != None:
            self.pairs_dict = pickle.load( open( filename, "rb" ) )
        elif data != None:
            self.pairs_dict = dict(data)

        #self.triplet_dict = self.make_triplets(pairs_dict, anchor_is_phrase, num_negative = 1)
        self.curr_index = 0

    def __len__(self):
        return len(self.triplets_caption)

    def __getitem__(self, index):
        triplet = self.triplets_caption[index]
        #triplet =  self.triplet_dict[index]
        return triplet

    def reset_counter(self):
        """
        Resets every epoch, since we avoid the incomplete batch
        """
        self.curr_index = 0

    def retrieve_embeddings(self):
        embedding_tuples = list(self.pairs_dict.values())
        dataset = [[],[]]
        lengths = [[],[]]
        num_tuples = len(embedding_tuples)

        for i in range(num_tuples):
            item = embedding_tuples[i]
            for type in range(2):
                dataset[type].append(item[type])
                lengths[type].append(item[type].shape[0])

        return dataset, lengths, num_tuples
    
    def get_dataset(self):
        #First is vids, second is captions
        dataset, lengths, num_tuples = self.retrieve_embeddings()

        # Sorted indices, first is vids, second is captions
        indices = [[],[]]

        datasets, indices = self.sort_pad_sequence(2, num_tuples, dataset, lengths, indices, True)

        return (dataset[1], dataset[0]), (indices[1], indices[0])

    def retrieve_triples(self, batch_size):
        # APN Examples. First is anchors, second is positives, third is negatives.
        examples = [[],[],[]]
        # APN Sequence lengths. First is anchors, second is positives, third is negatives.
        lengths = [[],[],[]]

        #Gets triplets, startaing at the first unused index. Num triplets
        #is batchsize.
        for i in range(self.curr_index, self.curr_index + batch_size):
            if (i >= self.__len__()):
                self.curr_index = 0
                break
            item = self.__getitem__(i)

            for example_type in range(3):
                example = item[example_type]
                examples[example_type].append(example)
                lengths[example_type].append(example.shape[0])
            self.curr_index += 1

        return examples, lengths

    def get_batch(self, batch_size):
        """
        Returns two tuples. The first is the processed anchors, positives, and negatives
        Elements within anchors, positives, and negativesare padded (for pytorch, using packed_padded sequence). Basically just a
        lot of pytorch jargon to get a padded batch for model input. The second tuple contains mappings 
        back to the original indices (gets sorted in decreasing size), for use later. 
        """
        # APN indices for the sorted sequences. First is A, second is P, third is N.
        indices=[[],[],[]]

        examples, lengths = self.retrieve_triples(batch_size)

        examples, indices = self.sort_pad_sequence(3, batch_size, examples, lengths, indices, False)

        return examples, indices

    # Pads sequences with zeros to make a square tensor.
    def pad_sequences(self, batch_size, example, length, index, backup):
        # Find max length sequence given the sorted lengths and examples
        max = length[0]
        if backup:
            padded = np.zeros((max, batch_size, example[0].shape[1]))
        else:
            padded = torch.zeros(max, batch_size, example[0].shape[1])

        # Effectively pads sequences with zeroes
        for i in range(batch_size):
            var = example[index[i]]
            if backup:
                padded[0:length[i], i, 0:example[0].shape[1]] = var
            else:
                padded[0:length[i], i, 0:example[0].shape[1]] = var.data

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
            examples[example_type] = nn.utils.rnn.pack_padded_sequence(examples[example_type], list(lengths[example_type]))

        return examples, indices

    # Triplet Margin Loss.
    def triplet_loss(self, A, P, N, margin=1.0):
        pos_dist = torch.norm(A-P).data
        neg_dist = torch.norm(A-N).data
        return float(pos_dist - neg_dist + margin)

    # Helper Function to swap anchor and positive examples along with their embeddings.
    def swap(self, A, P, A_embedding, P_embedding):
        temp = A
        anchor = P
        positive = temp
        temp = A_embedding
        anchor_embedding = P_embedding
        positive_embedding = temp

        return anchor, positive, anchor_embedding, positive_embedding

    def mine_triplets_all(self, embedding_tuples):
        captions = [[],[]]

        #Tuples of inputs and outputs - First is clips, second is captions
        inputs = (self.vids_backup, self.words_backup)
        outputs = (embedding_tuples[1], embedding_tuples[0])

        # Loop over captions
        for index in range(inputs[1].shape[1]):
            #Create anchor and positive from pairs
            anchor = inputs[1][:,index,:]
            anchor_embedding = outputs[1][index, :]
            positive = inputs[0][:,index,:]
            positive_embedding = outputs[0][index, :]

            # Loop over clips
            for neg_index in range(inputs[0].shape[1]):
                # Check both possible triples for positive loss
                for anchor_type in range(2):
                    negative = inputs[anchor_type][:,neg_index,:]
                    negative_embedding = outputs[anchor_type][neg_index]
                    if self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding) > 0:
                        captions[anchor_type].append((anchor.squeeze(), positive.squeeze(), negative.squeeze()))

                    anchor, positive, anchor_embedding, positive_embedding = self.swap(anchor, positive, anchor_embedding, positive_embedding)

        self.triplets_caption = captions[0]
        self.triplets_clips = captions[1]

        return len(captions[0]), len(captions[1])