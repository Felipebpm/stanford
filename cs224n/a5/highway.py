#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    ### YOUR CODE HERE for part 1f
    def __init__(self, embed_size):
        """ Init Highway Module.

        @param embed_size (int): size of word embeddings
        """
        super(Highway, self).__init__()
        self.projection = nn.Linear(embed_size, embed_size, bias=True)
        self.gate = nn.Linear(embed_size, embed_size, bias=True)

    def forward(self, conv_out):
        """ Forward pass of character decoder.

        @param conv_out (Tensor): tensor of floats, shape (batch_size, embedding_size)

        @returns x_highway (Tensor): tensor of floats, shape (batch_size, embedding_size)
        """
        x_proj = torch.relu(self.projection(conv_out))
        x_gate = torch.sigmoid(self.gate(conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * conv_out

        return x_highway
    ### END YOUR CODE

