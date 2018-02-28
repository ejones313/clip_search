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


    def get_batch(self, batch_size):
        """
        Returns two tuples. The first is the processed anchors, positives, and negatives
        Elements within anchors, positives, and negativesare padded (for pytorch, using packed_padded sequence). Basically just a
        lot of pytorch jargon to get a padded batch for model input. The second tuple contains mappings 
        back to the original indices (gets sorted in decreasing size), for use later. 
        """
        anchors = []
        positives = []
        negatives = []
        anchor_lengths = []
        positive_lengths = []
        negative_lengths = []

        #Gets triplets, startaing at the first unused index. Num triplets
        #is batchsize. 
        for i in range(self.curr_index, self.curr_index + batch_size):
            if (i >= self.__len__()):
                self.curr_index = 0
                break
            item = self.__getitem__(i)

            anchors.append(item[0])
            positives.append(item[1])
            negatives.append(item[2])

            anchor_lengths.append(item[0].shape[0])
            positive_lengths.append(item[1].shape[0])
            negative_lengths.append(item[2].shape[0])
            self.curr_index += 1

        #For pytorch, sorts the components of the triples by the length of the sequence (will be unsorted correctly later)
        anchor_lengths, anchor_indices = torch.sort(torch.IntTensor(anchor_lengths), descending = True)
        positive_lengths, positive_indices = torch.sort(torch.IntTensor(positive_lengths), descending = True)
        negative_lengths, negative_indices = torch.sort(torch.IntTensor(negative_lengths), descending = True)

        #Initializes array to copy things with different lengths to (and thus to pad)
        max_anchor = anchor_lengths[0]
        max_positive = positive_lengths[0]
        max_negative = negative_lengths[0]

        #anchor_padded = np.zeros((max_anchor, batch_size, anchors[0].shape[1]))
        #positive_padded = np.zeros((max_positive, batch_size, positives[0].shape[1]))
        #negative_padded = np.zeros((max_negative, batch_size, negatives[0].shape[1]))
        anchor_padded = torch.zeros(max_anchor, batch_size, anchors[0].shape[1])
        positive_padded = torch.zeros(max_positive, batch_size, positives[0].shape[1])
        negative_padded = torch.zeros(max_negative, batch_size, negatives[0].shape[1])

        #Effectively pads sequences with zeroes
        for i in range(batch_size):
            #print(anchors[0].shape[1])
            #print(positives[0].shape[1])
            #print(negatives[0].shape[1])
            #print(anchors[anchor_indices[0]].shape)
            #print(positives[positive_indices[0]].shape)
            #print(negatives[negative_indices[0]].shape)
            #pause = input("wait")
            print(anchors[0].shape[1])
            print(anchor_lengths)
            anchor_padded[0:anchor_lengths[i], i, 0:anchors[0].shape[1]] = anchors[anchor_indices[i]].data
            positive_padded[0:positive_lengths[i], i, 0:positives[0].shape[1]] = positives[positive_indices[i]].data
            negative_padded[0:negative_lengths[i], i, 0:negatives[0].shape[1]] = negatives[negative_indices[i]].data


        #Converts to variables
        #anchors = Variable(torch.from_numpy(np.array(anchor_padded)).float())
        #positives = Variable(torch.from_numpy(np.array(positive_padded)).float())
        #negatives = Variable(torch.from_numpy(np.array(negative_padded)).float())
        anchors = Variable(anchor_padded.float())
        positives = Variable(positive_padded.float())
        negatives = Variable(negative_padded.float())

        #Obnoxious pytorch thing
        anchors = nn.utils.rnn.pack_padded_sequence(anchors, list(anchor_lengths))
        positives = nn.utils.rnn.pack_padded_sequence(positives, list(positive_lengths))
        negatives = nn.utils.rnn.pack_padded_sequence(negatives, list(negative_lengths))

        return (anchors, positives, negatives), (anchor_indices, positive_indices, negative_indices)




    def triplet_loss(self, A, P, N, margin=1.0):
        #pos_dist = np.linalg.norm(A-P)
        #neg_dist = np.linalg.norm(A-N)
        pos_dist = torch.norm(A-P).data
        neg_dist = torch.norm(A-N).data
        return float(pos_dist - neg_dist + margin)



    def mine_triplets_all(self, embedding_tuples):
        triplets_caption = []
        triplets_clips = []
        captions_out = embedding_tuples[0]
        clips_out = embedding_tuples[1]
        captions_in = self.words_backup
        clips_in = self.vids_backup

        for index in range(captions_in.shape[1]):
            anchor = captions_in[:,index,:]
            anchor_embedding = captions_out[index]
            positive = clips_in[:,index,:]
            positive_embedding = clips_out[index]

            for neg_index in range(clips_in.shape[1]):
                negative = clips_in[:,neg_index,:]
                negative_embedding = clips_out[neg_index]
                if self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding) > 0:
                    #Caption is anchor
                    triplets_caption.append((anchor.squeeze(), positive.squeeze(), negative.squeeze()))

                temp = anchor
                anchor = positive
                positive = temp
                temp = anchor_embedding
                anchor_embedding = positive_embedding
                positive_embedding = temp

                negative = captions_in[:,neg_index,:]
                negative_embedding = captions_out[neg_index]
                if self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding) > 0:
                    #Clip is anchor
                    triplets_clips.append((anchor.squeeze(), positive.squeeze(), negative.squeeze()))

        self.triplets_caption = triplets_caption
        self.triplets_clips = triplets_clips

        return len(triplets_caption), len(triplets_clips)