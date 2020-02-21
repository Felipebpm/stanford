#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    ### YOUR CODE HERE for part 1g
    def __init__(self, kernel_size, num_channels, embed_size):
        """ Init CNN Module.

        @param kernel_size (int): kernel size for the convolution
        @param num_channels (int): number of channels for convolution
        @param embed_size (int): size of character embeddings
        """
        super(CNN, self).__init__()
        self.convolution = nn.Conv1d(embed_size, num_channels, kernel_size, padding=1, bias=True)

    def forward(self, x_reshaped):
        """ Forward pass of character decoder.

        @param x_reshaped (Tensor): tensor of floats, shape (batch_size, 
        character embedding size, max_word_len_size)

        @returns x_conv_out (Tensor): tensor of floats, shape (batch_size, word embeddingsize)
        """
        x_conv = self.convolution(x_reshaped)
        x_conv_out = torch.max(torch.relu(x_conv), 2).values

        return x_conv_out
    ### END YOUR CODE

