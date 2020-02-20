#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    ### YOUR CODE HERE for part 1g
    def __init__(self, embed_char, embed_word):
        super(CNN, self).__init__()
        self.kernel_size = 5
        self.padding_size = 1
        self.convolution = nn.Conv1d(embed_char, embed_word, self.kernel_size, padding = self.padding_size)

    def forward(self, x):
        x_conv = self.convolution(x)
        x_conv_out = torch.max(F.relu(x_conv), 2)[0]
        return x_conv_out
    ### END YOUR CODE

