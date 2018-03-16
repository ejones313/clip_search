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
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn
from torch.autograd import Variable
import numpy as np

import utils
import net
import data_prep

from datetime import datetime

from valid import validate_L2_V2


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
    word_model.train()
    vid_model.train()
    word_model.zero_grad()
    vid_model.zero_grad()

    batch_size = params.batch_size
    num_batches = dataSet.pairs_len() // batch_size

    total_loss = 0

    #Iterate through all batches except the incomplete one.
    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = (batch_num + 1) * batch_size

        packed_dataset, indices = dataSet.get_pairs(start, end)

        words = packed_dataset[0]
        vids = packed_dataset[1]
        word_indices = indices[0]
        video_indices = indices[1]
        word_output = word_model(words)
        video_output = vid_model(vids)

        # Undo pack_padded_sequence
        word_output, word_lengths = nn.utils.rnn.pad_packed_sequence(word_output)
        video_output, video_lengths = nn.utils.rnn.pad_packed_sequence(video_output)

        # Unscramble output
        word_order = torch.zeros(word_indices.shape).long()
        vid_order = torch.zeros(video_indices.shape).long()
        for i in range(word_indices.size()[0]):
            word_order[word_indices[i]] = i
            vid_order[video_indices[i]] = i

        word_unscrambled = utils.unscramble(word_output, word_lengths, word_order, batch_size, params.cuda)
        video_unscrambled = utils.unscramble(video_output, video_lengths, vid_order, batch_size, params.cuda)

        if params.cuda:
            loss = Variable(torch.FloatTensor([0]).cuda(), requires_grad=True)
        else:
            loss = Variable(torch.FloatTensor([0]), requires_grad=True)

        for triplet_type in range(1):
            for anchor_num in range(batch_size):
                if triplet_type == 0:
                    A = torch.unsqueeze(word_unscrambled[anchor_num,:], 0).expand(batch_size, -1)
                    P = torch.unsqueeze(video_unscrambled[anchor_num,:], 0).expand(batch_size, -1)
                    N = video_unscrambled
                else:
                    A = torch.unsqueeze(video_unscrambled[anchor_num,:], 0).expand(batch_size, -1)
                    P = torch.unsqueeze(word_unscrambled[anchor_num,:], 0).expand(batch_size, -1)
                    N = word_unscrambled
                loss = loss + loss_fn(A,P,N)
        
        print('Batch: %d' % batch_num)
        print(loss.data[0])
            
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




def train_and_evaluate(models, optimizers, filenames, loss_fn, params, anchor_is_phrase = True):
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
    train_dataset = data_prep.Dataset(filename = train_filename, anchor_is_phrase = anchor_is_phrase, cuda = params.cuda)
    val_dataset = data_prep.Dataset(filename = val_filename, anchor_is_phrase = anchor_is_phrase, cuda = params.cuda)

    best_val = float("-inf")
    best_dist_diff = -1
    #Train
    start_time = datetime.now()
    for epoch in range(params.num_epochs):
        is_best = False
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        print("Starting epoch: {}. Time elapsed: {}".format(epoch+1, str(datetime.now()-start_time)))
        train_loss = train(word_model, vid_model, word_optimizer, vid_optimizer, loss_fn, train_dataset, params)[0]

        things, indices = val_dataset.get_pairs(0, val_dataset.pairs_len())
        avg_prctile, dist_diff = validate_L2_V2(word_model, vid_model, things, indices, cuda = params.cuda)
        print("Train Loss: {}, Val Scores: (Pos_prctile: {}, Pos_dist-Neg_dist: {})\n".format(train_loss, avg_prctile, dist_diff))

        if avg_prctile > best_val:
            best_val = avg_prctile
            best_dist_diff = dist_diff
            is_best = True

        utils.save_checkpoint({'epoch': epoch +1,
                                'word_state_dict': word_model.state_dict(),
                                'vid_state_dict': vid_model.state_dict(),
                                'word_optim_dict': word_optimizer.state_dict(),
                                'vid_optim_dict': vid_optimizer.state_dict(),
                                'val_avg_prctile': avg_prctile,
                                'val_dist_diff': dist_diff,
                                'train_loss': train_loss}, is_best =is_best,
                                checkpoint="weights_and_val")
    return(best_val, best_dist_diff)

def main(params, args):
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
    filenames["train"] = params.train_file
    filenames["val"] = params.val_file
    logging.info("- done.")

    # Define the models and optimizers
    models = {}
    word_model = net.Net(params, True).cuda() if params.cuda else net.Net(params, True)
    vid_model = net.Net(params, False).cuda() if params.cuda else net.Net(params, False)
    models["word"] = word_model
    models["vid"] = vid_model

    optimizers = {}
    word_optimizer = optim.Adadelta(word_model.parameters(), lr = params.lr, weight_decay = params.reg_strength)
    vid_optimizer = optim.Adadelta(vid_model.parameters(), lr = params.lr, weight_decay = params.reg_strength)
    optimizers["word"] = word_optimizer
    optimizers["vid"] = vid_optimizer

    # fetch loss function and metrics
    loss_fn = torch.nn.modules.loss.TripletMarginLoss(margin=params.margin)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    return(train_and_evaluate(models, optimizers, filenames, loss_fn, params))


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    args.model_dir = '.'
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    main(params, args)