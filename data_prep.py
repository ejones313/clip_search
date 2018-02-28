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


    def make_triplets(self, pairs_dict, anchor_is_phrase, num_negative = 3):
        """
        Mechanism for naievely constructing triplets. Pairs_dict is our loaded data,
        in dictionary form, containing values of (video, caption, vid_id). Anchor_is_phrase
        determines whether or not the anchor is the caption of the word. Based on this variable,
        constructs num_negative triples of (anchor, corresponding positive, corresponding negative)
        for each anchor in the dataset. Return value is a dictionary with nonnegative integer
        keys and values corresponding to the triple. 
        """
        triplet_dict = {}
        counter = 0
        for key in pairs_dict:
            video, caption, vid_id = pairs_dict[key]
            for j in range(num_negative):
                rand_key = random.choice(list(pairs_dict.keys()))
                while rand_key == key:
                    rand_key = random.choice(list(pairs_dict.keys()))
                rvideo, rcaption, rvid_id = pairs_dict[rand_key]
                triple = None
                if anchor_is_phrase:
                    triple = (caption, video, rvideo)
                else:
                    triple = (video, caption, rcaption)
                triplet_dict[counter] = triple
                counter += 1
        return triplet_dict

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

    
    def get_dataset(self):
        embedding_tuples = list(self.pairs_dict.values())
        words = []
        vids = []
        word_lengths = []
        vid_lengths = []
        for i in range(len(embedding_tuples)):
            item = embedding_tuples[i]
            vids.append(item[0])
            words.append(item[1])
            vid_lengths.append(item[0].shape[0])
            word_lengths.append(item[1].shape[0])

        #For pytorch, sorts the components of the triples by the length of the sequence (will be unsorted correctly later)
        word_lengths, word_indices = torch.sort(torch.IntTensor(word_lengths), descending = True)
        vid_lengths, vid_indices = torch.sort(torch.IntTensor(vid_lengths), descending = True)
        
        #Initializes array to copy things with different lengths to (and thus to pad)
        max_word = word_lengths[0]
        max_vid = vid_lengths[0]
        
        word_padded = np.zeros((max_word, len(embedding_tuples), words[0].shape[1]))
        vid_padded = np.zeros((max_vid, len(embedding_tuples), vids[0].shape[1]))
        
        #Effectively pads sequences with zeroes
        for i in range(len(embedding_tuples)):
            word_padded[0:word_lengths[i], i, 0:words[0].shape[1]] = words[word_indices[i]]
            vid_padded[0:vid_lengths[i], i, 0:vids[0].shape[1]] = vids[vid_indices[i]]
            
        #Converts to variables
        words = Variable(torch.from_numpy(np.array(word_padded)).float())
        vids = Variable(torch.from_numpy(np.array(vid_padded)).float())

        self.words_backup = words.clone()
        self.vids_backup = vids.clone()
        
        #Obnoxious pytorch thing
        words = nn.utils.rnn.pack_padded_sequence(words, list(word_lengths))
        vids = nn.utils.rnn.pack_padded_sequence(vids, list(vid_lengths))
        
        return (words, vids), (word_indices, vid_indices)

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