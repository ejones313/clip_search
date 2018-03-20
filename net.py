"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, params, anchor_is_phrase):
        """
        Simple LSTM, used to generate the LSTM for both the word and video
        embeddings. 
        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
            is_phrase: is word lstm or the vid lstm
        """
        super(Net, self).__init__()
        if anchor_is_phrase:
            self.lstm = nn.GRU(params.word_embedding_dim, params.hidden_dim, 1)#, batch_first=True)
        else:
            self.lstm = nn.GRU(params.vid_embedding_dim, params.hidden_dim, 1)#, batch_first=True)

    def forward(self, s, anchor_is_phrase = False):
        """
        Forward prop. 
        """
        s, _ = self.lstm(s)
        s.data.contiguous()
        return s