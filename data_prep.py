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
        #First is vids, second is tuples
        dataset, lengths, num_tuples = self.retrieve_embeddings()
        indices = [[],[]]

        for type in range(2):
            #For pytorch, sorts the components of the triples by the length of the sequence (will be unsorted correctly later)
            lengths[type], indices[type] = torch.sort(torch.IntTensor(lengths[type]), descending = True)
        
            #Initializes array to copy things with different lengths to (and thus to pad)
            max = lengths[type][0]
            padded = np.zeros((max, num_tuples, dataset[type][0].shape[1]))
        
            #Effectively pads sequences with zeroes
            for i in range(num_tuples):
                padded[0:lengths[type][i], i, 0:dataset[type][0].shape[1]] = dataset[type][indices[type][i]]
            
            #Converts to variables
            dataset[type] = Variable(torch.from_numpy(np.array(padded)).float())

            if type == 1:
                self.words_backup = dataset[1].clone()
            else:
                self.vids_backup = dataset[0].clone()

            # Obnoxious pytorch thing
            dataset[type] = nn.utils.rnn.pack_padded_sequence(dataset[type], list(lengths[type]))

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

        for example_type in range(3):
            # For pytorch, sorts the components of the triples by the length of the sequence (will be unsorted correctly later)
            lengths[example_type], indices[example_type] = torch.sort(torch.IntTensor(lengths[example_type]), descending=True)

            max = lengths[example_type][0]
            padded = torch.zeros(max, batch_size, examples[example_type][0].shape[1])

            # Effectively pads sequences with zeroes
            for i in range(batch_size):
                padded[0:lengths[example_type][i], i, 0:examples[example_type][0].shape[1]] = examples[example_type][indices[example_type][i]].data

            # Convert to Variables
            examples[example_type] = Variable(padded.float())

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