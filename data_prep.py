import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import pickle

class Dataset():

    def __init__(self, filename = None, anchor_is_phrase = True, data = None):
        if filename != None:
            self.pairs_dict = pickle.load( open( filename, "rb" ) )
        elif data != None:
            self.pairs_dict = dict(data)

        self.curr_index_word = 0
        self.curr_index_vid = 0

    def len(self):
        return len(self.triplets_caption)

    def getitem(self, index, anchor_is_phrase):
        if anchor_is_phrase:
            triplet = self.triplets_caption[index]
        else:
            triplet = self.triplets_clips[index]
        return triplet

    def reset_counter(self):
        """
        Resets every epoch, since we avoid the incomplete batch
        """
        self.curr_index_word = 0
        self.curr_index_vid = 0

    def retrieve_embeddings(self):
        embedding_tuples = list(self.pairs_dict.values())

        # Pairs of positive examples. First is videos, second is captions.
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
        #First is vids, second is captions in dataset
        dataset, lengths, num_tuples = self.retrieve_embeddings()

        # Sorted indices, first is vids, second is captions
        indices = [[],[]]

        datasets, indices = self.sort_pad_sequence(2, num_tuples, dataset, lengths, indices, True)

        return (dataset[1], dataset[0]), (indices[1], indices[0])

    def retrieve_triples(self, batch_size, anchor_is_phrase):
        # APN Examples. First is anchors, second is positives, third is negatives.
        examples = [[],[],[]]
        # APN Sequence lengths. First is anchors, second is positives, third is negatives.
        lengths = [[],[],[]]

        curr_index = self.curr_index_vid
        if anchor_is_phrase:
            curr_index = self.curr_index_word

        #Gets triplets, startaing at the first unused index. Num triplets
        #is batchsize.
        for i in range(curr_index, curr_index + batch_size):
            if (i >= self.len()):
                raise Exception("Exceeded batch length")

            item = self.getitem(i, anchor_is_phrase)
            if anchor_is_phrase:
                item_lengths = self.triplets_caption_lengths[i]
            else:
                item_lengths = self.triplets_clips_lengths[i]

            for example_type in range(3):
                example = item[example_type]
                examples[example_type].append(example)
                lengths[example_type].append(item_lengths[example_type])

        if anchor_is_phrase:
            self.curr_index_word += batch_size
        else:
            self.curr_index_vid += batch_size

        return examples, lengths

    def get_batch(self, batch_size, anchor_is_phrase):
        """
        Returns two tuples. The first is the processed anchors, positives, and negatives
        Elements within anchors, positives, and negativesare padded (for pytorch, using packed_padded sequence). Basically just a
        lot of pytorch jargon to get a padded batch for model input. The second tuple contains mappings 
        back to the original indices (gets sorted in decreasing size), for use later. 
        """

        # APN indices for the sorted sequences. First is A, second is P, third is N.
        indices=[[],[],[]]

        examples, lengths = self.retrieve_triples(batch_size, anchor_is_phrase)

        examples, indices = self.sort_pad_sequence(3, batch_size, examples, lengths, indices, False)

        return examples, indices

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

            padded = torch.zeros(lengths[example_type][0], batch_size, examples[example_type][0].shape[1])

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

    def shuffle_together(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def save_triplets(self, triplets, lengths):
        self.shuffle_together(triplets[0], lengths[0])
        self.shuffle_together(triplets[1], lengths[1])

        self.triplets_caption = triplets[0]
        self.triplets_clips = triplets[1]

        self.triplets_caption_lengths = lengths[0]
        self.triplets_clips_lengths = lengths[1]

    def mine_triplets_all(self, embedding_tuples, lengths_tuple):
        # DON'T THINK THIS IS CORRECTLY MAKING THE VIDEO ANCHOR TRIPLETS RIGHT NOW

        triplets = [[],[]]
        lengths = [[],[]]

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
                if index != neg_index:
                    # Check both possible triples for positive loss
                    for anchor_type in range(2):
                        negative = inputs[anchor_type][:,neg_index,:]
                        negative_embedding = outputs[anchor_type][neg_index]

                        if self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding, margin = 0.2) > 0:
                            triplets[anchor_type].append((anchor.squeeze(), positive.squeeze(), negative.squeeze()))
                            lengths[anchor_type].append((lengths_tuple[anchor_type][index], lengths_tuple[1-anchor_type][index], lengths_tuple[1-anchor_type][neg_index]))
                            
                        anchor, positive, anchor_embedding, positive_embedding = self.swap(anchor, positive, anchor_embedding, positive_embedding)

        self.save_triplets(triplets, lengths)

        return len(triplets[0]), len(triplets[1])

