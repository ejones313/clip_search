"""Train the model"""

"""
QUESTIONS:
    -Train: 10000, Val: 1000
    -Downsampling
    -When to regenerate triplets
"""

import argparse
import os

import logging
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn
#from tqdm import trange

import utils
import net
import data_prep
from torch.autograd import Variable
import pickle
import math

from datetime import datetime

from valid import validate


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(word_model, vid_model, word_optimizer, vid_optimizer, loss_fn, dataSet, params):
    """ Does gradient descent on one epoch
    Args:
        word_model: (torch.nn.Module) the LSTM for word embeddings
        vid_model: (torch.nn.Module) the LSTM for video embeddings
        word_optimizer: (torch.optim) optimizer for parameters of word_model
        vid_optimizer: (torch.optim) optimizer for parameters of vid_model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataSet: (Dataset class) custom dataset class containing the data
        params: (Params) hyperparameters
        anchor_is_phrase: (bool) Defines the anchor (true is phrase, false is clip)
    """

    # set model to training mode.
    loss_var = 0
    word_model.train()
    vid_model.train()
    word_model.zero_grad()
    vid_model.zero_grad()

    #Calculate number of batches
    #batch_size = params.batch_size
    #dataset_size = len(dataSet)
    #num_batches = dataset_size // batch_size

    #
    packed_dataset, indices = dataSet.get_dataset()
    words = packed_dataset[0]
    vids = packed_dataset[1]
    word_indices = indices[0]
    video_indices = indices[1]
    word_output = word_model(words)
    video_output = vid_model(vids)

    #Undo pack_padded_sequence
    word_output, word_lengths = nn.utils.rnn.pad_packed_sequence(word_output)
    video_output, video_lengths = nn.utils.rnn.pad_packed_sequence(video_output)

    #Unscramble output, and unpad
    word_unscrambled = utils.unscramble(word_output, word_lengths, word_indices, len(word_indices), cuda = params.cuda)
    video_unscrambled = utils.unscramble(video_output, video_lengths, video_indices, len(word_indices), cuda = params.cuda)

    batch_size = params.batch_size
    num_batches = len(word_lengths) // batch_size

    total_loss = 0
    
    #Iterate through all batches except the incomplete one.
    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = (batch_num + 1) * batch_size
        batches, idx = dataSet.mine_triplets_all((word_unscrambled[start:end], video_unscrambled[start:end]),
                                                                        (word_lengths[start:end], video_lengths[start:end]))
        for anchor_type in [0, 1]:
            batch, indices = batches[anchor_type], idx[anchor_type]

            anchor_batch =  batch[0]
            positive_batch = batch[1]
            negative_batch = batch[2]

            anchor_indices = indices[0]
            positive_indices = indices[1]
            negative_indices = indices[2]

            # compute model output and loss, putting each component of the batch into the appropriate LSTM
            if anchor_type:
                anchor_output = word_model(anchor_batch)
                positive_output = vid_model(positive_batch)
                negative_output = vid_model(negative_batch)
            else:
                anchor_output = vid_model(anchor_batch)
                positive_output = word_model(positive_batch)
                negative_output = word_model(negative_batch)


            #Undo pack_padded_sequence
            anchor_output, anchor_lengths = nn.utils.rnn.pad_packed_sequence(anchor_output)
            positive_output, positive_lengths = nn.utils.rnn.pad_packed_sequence(positive_output)
            negative_output, negative_lengths = nn.utils.rnn.pad_packed_sequence(negative_output)

            #Unscramble output, and unpad
            anchor_unscrambled = utils.unscramble(anchor_output, anchor_lengths, anchor_indices, batch_size, cuda = params.cuda)
            positive_unscrambled = utils.unscramble(positive_output, positive_lengths, positive_indices, batch_size, cuda = params.cuda)
            negative_unscrambled = utils.unscramble(negative_output, negative_lengths, negative_indices, batch_size, cuda = params.cuda)

            #Compute loss over the batch
            loss = loss_fn(anchor_unscrambled, positive_unscrambled, negative_unscrambled)

            # clear previous gradients, compute gradients of all variables wrt loss
            vid_optimizer.zero_grad()
            word_optimizer.zero_grad()
            #Backprop
            loss.backward()

            # performs updates using calculated gradients
            word_optimizer.step()
            vid_optimizer.step()

            total_loss += loss.data

    return total_loss


def train_and_evaluate(models, optimizers, filenames, loss_fn, params, anchor_is_phrase = True, subset_size = 50):
    """Train the model over many epochs
    Args:
        word_model: (torch.nn.Module) the LSTM for word embeddings
        vid_model: (torch.nn.Module) the LSTM for video embeddings
        train_filename: (string) filename (pkl) for our training dataset
        word_optimizer: (torch.optim) optimizer for parameters of word_model
        vid_optimizer: (torch.optim) optimizer for parameters of vid_model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataSet: (Dataset class) custom dataset class containing the data
        params: (Params) hyperparameters
        anchor_is_phrase: (bool) Defines the anchor (true is phrase, false is clip)
    """
    # reload weights from restore_file if specified
    #if restore_file is not None:
    #    restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
    #    logging.info("Restoring parameters from {}".format(restore_path))
    #    utils.load_checkpoint(restore_path, model, optimizer)
    
    word_model = models["word"]
    vid_model = models["vid"]
    word_optimizer = optimizers["word"]
    vid_optimizer = optimizers["vid"]
    train_filename = filenames["train"]
    val_filename = filenames["val"]


    #Load train dataset
    full_dataset = pickle.load( open( train_filename, "rb" ) )
    val_dataset = data_prep.Dataset(filename = val_filename, anchor_is_phrase = anchor_is_phrase, cuda = params.cuda)
    tuple_list = full_dataset.items()
    datasets = []
    for i in range(math.floor(len(tuple_list)/subset_size)):
        datasets.append(data_prep.Dataset(data = list(tuple_list)[subset_size*i:(i+1)*subset_size], cuda = params.cuda))

    train_losses = []
    best_val = float("-inf")
    #Train
    start_time = datetime.now()
    for epoch in range(params.num_epochs):
        is_best = False
        print("Starting epoch: {}. Time elapsed: {}".format(epoch, str(datetime.now()-start_time)))
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        for dataset in datasets:
            train_loss = train(word_model, vid_model, word_optimizer, vid_optimizer, loss_fn, dataset, params)
            train_losses.append(train_loss)
            dataset.reset_counter()

        # SAVE MODEL PARAMETERS AND VALIDATION PERFORMANCE
        val_scores = validate(word_model, vid_model, val_dataset, cuda = params.cuda)
        print("Train Loss: {}, Val Scores: Vid good: {} Word good: {} Total good {}".format(sum(train_losses)/len(train_losses), val_scores[0], val_scores[1], val_scores[2]))
        
        if val_scores[2] > best_val:
            best_val = val_scores[2]
            is_best = True

        utils.save_checkpoint({'epoch': epoch +1,
                                'word_state_dict': word_model.state_dict(),
                                'vid_state_dict': vid_model.state_dict(),
                                'word_optim_dict': word_optimizer.state_dict(),
                                'vid_optim_dict': vid_optimizer.state_dict(),
                                'val_scores': val_scores,
                                'train_losses': train_losses}, is_best =is_best,
                                checkpoint="weights_and_val")

def triplet_loss_all(triplets, margin=1.0):
    triplets = triplets.cuda()

    ap_distances = (triplets[:, 0] - triplets[:, 1]).pow(2).sum(1).pow(.5) #remove square root?
    an_distances = (triplets[:, 0] - triplets[:, 2]).pow(2).sum(1).pow(.5)
    losses = F.relu(ap_distances - an_distances + margin)

    return losses.mean()

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    args.model_dir = '.' 
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    #Set LSTM hyperparameters
    params.word_embedding_dim = 300
    params.word_hidden_dim = 600
    params.vid_embedding_dim = 500
    params.vid_hidden_dim = 600
    params.batch_size = 5

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    # load data
    filenames = {}
    filenames["train"] = 'subset.pkl'
    filenames["val"] = 'val_500.pkl'
    logging.info("- done.")

    # Define the models and optimizers
    models = {}
    word_model = net.Net(params, True).cuda() if params.cuda else net.Net(params, True)
    vid_model = net.Net(params, False).cuda() if params.cuda else net.Net(params, False)
    models["word"] = word_model
    models["vid"] = vid_model

    optimizers = {}
    word_optimizer = optim.Adam(word_model.parameters(), lr=params.learning_rate)
    vid_optimizer = optim.Adam(vid_model.parameters(), lr=params.learning_rate)
    optimizers["word"] = word_optimizer
    optimizers["vid"] = vid_optimizer
    
    # fetch loss function and metrics
    loss_fn = triplet_loss_all(margin=0.2)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(models, optimizers, filenames, loss_fn, params, subset_size = 10)

