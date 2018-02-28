"""Train the model"""

import argparse
import os

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#from tqdm import trange

import utils
import net
import data_prep
from torch.autograd import Variable
import pickle
import math


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


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


def train(word_model, vid_model, word_optimizer, vid_optimizer, loss_fn, dataSet, params, anchor_is_phrase):
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
    word_unscrambled = unscramble(word_output, word_lengths, word_indices, len(word_indices))
    video_unscrambled = unscramble(video_output, video_lengths, video_indices, len(word_indices))

    num_triplets, _ = dataSet.mine_triplets_all((word_unscrambled, video_unscrambled))

    batch_size = params.batch_size
    num_batches = num_triplets // batch_size
    
    #Iterate through all batches except the incomplete one.
    for batch_num in range(0,num_batches-1):
        batch, indices = dataSet.get_batch(batch_size)

        anchor_batch = batch[0]
        positive_batch = batch[1]
        negative_batch = batch[2]

        anchor_indices = indices[0]
        positive_indices = indices[1]
        negative_indices = indices[2]

        # compute model output and loss, putting each component of the batch into the appropriate LSTM
        if anchor_is_phrase:
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
        anchor_unscrambled = unscramble(anchor_output, anchor_lengths, anchor_indices, batch_size)
        positive_unscrambled = unscramble(positive_output, positive_lengths, positive_indices, batch_size)
        negative_unscrambled = unscramble(negative_output, negative_lengths, negative_indices, batch_size)

        #Compute loss over the batch
        loss = loss_fn(anchor_unscrambled, positive_unscrambled, negative_unscrambled)

        print("Loss", loss.data)

        # clear previous gradients, compute gradients of all variables wrt loss
        vid_optimizer.zero_grad()
        word_optimizer.zero_grad()
        #Backprop
        loss.backward()

        # performs updates using calculated gradients
        word_optimizer.step()
        vid_optimizer.step()


def train_and_evaluate(word_model, vid_model, train_filename, word_optimizer, vid_optimizer, loss_fn, params, anchor_is_phrase = True, subset_size = 50):
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
        anchor_is_prhase: (bool) Defines the anchor (true is phrase, false is clip)
    """
    # reload weights from restore_file if specified
    #if restore_file is not None:
    #    restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
    #    logging.info("Restoring parameters from {}".format(restore_path))
    #    utils.load_checkpoint(restore_path, model, optimizer)
    
    #Load train dataset
    full_dataset = pickle.load( open( train_filename, "rb" ) )
    tuple_list = full_dataset.items()
    datasets = []
    for i in range(math.floor(len(tuple_list)/subset_size)):
        # Add case for subset size not divisible by length
        datasets.append(data_prep.Dataset(data = list(tuple_list)[subset_size*i:(i+1)*subset_size]))
    #train_dataset = data_prep.Dataset(train_filename)

    #Train
    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        for dataset in datasets:
            print('New subepoch')
            train(word_model, vid_model, word_optimizer, vid_optimizer, loss_fn, dataset, params, anchor_is_phrase)
            train_dataset.reset_counter()
    

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
    params.batch_size = 4

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
    train_filename = 'subset.pkl'
    logging.info("- done.")

    # Define the models and optimizers
    word_model = net.Net(params, True).cuda() if params.cuda else net.Net(params, True)
    vid_model = net.Net(params, False).cuda() if params.cuda else net.Net(params, False)
    word_optimizer = optim.Adam(word_model.parameters(), lr=params.learning_rate)
    vid_optimizer = optim.Adam(vid_model.parameters(), lr=params.learning_rate)
    
    # fetch loss function and metrics
    loss_fn = torch.nn.modules.loss.TripletMarginLoss()

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(word_model, vid_model, train_filename, word_optimizer, vid_optimizer, loss_fn, params)