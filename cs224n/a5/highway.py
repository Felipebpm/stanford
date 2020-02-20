#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    ### YOUR CODE HERE for part 1f
    def __init__(self, embedding_size):
        super(Highway, self).__init__()
        self.linear_proj = nn.Linear(embedding_size, embedding_size)
        self.linear_gate = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        x_proj = F.relu(self.linear_proj(x))
        x_gate = torch.sigmoid(self.linear_gate(x))
        x_highway = x_gate * x_proj + (1 - x_gate) * x
        return x_highway





    ### END YOUR CODE

