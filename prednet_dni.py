# -*- coding: utf-8 -*-

"""
PredNet in PyTorch.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from prednet import PredNet


class PredNetDNI(nn.Module):
    """
    PredNet realized by zcr and modified by arewellborn.
    """

    def __init__(
        self, prednet,
    ):
        super(PredNetDNI, self).__init__()
        self.prednet = prednet
        self.num_layers = prednet.num_layers
        self.row_axis = prednet.row_axis
        self.col_axis = prednet.col_axis
        self.get_initial_states = prednet.get_initial_states

        # Freeze all of the existing PredNet layers
        for param in self.prednet.parameters():
            param.requires_grad = False

        # Spatially average pool lower layers to match upper layer dims
        self.pool_list = nn.ModuleList()
        for n in range(1, 4):
            self.pool_list.append(nn.AvgPool2d(kernel_size=2 ** n, stride=2 ** n))

        # We are concatenating and flattening all elements for the out gate for all layers
        in_features = (
            7 * (3 + 48 + 96 + 192) * 30 * 30
        )  # batch * all layer features * (h / 2 ^ 3) * (w / 2 ^ 3)
        self.linear_layer = nn.Linear(in_features=in_features, out_features=1)

    def forward(self, A0_withTimeStep, initial_states):

        # hidden_states will return an array of shape (1, batch size) representing the last time step for each sequence
        output, hidden_states = self.prednet(A0_withTimeStep, initial_states)

        # Get only R_l layers from hidden_states (first batch of states)
        r_layers = hidden_states[: self.num_layers]

        # Concat out gate layers across channel/feature axis after pooling
        output = []
        for i, layer in enumerate(reversed(r_layers)):
            if i == 0:
                output.append(layer)
            else:
                pool = self.pool_list[i - 1]
                output.append(pool(layer))

        output = torch.cat(output, dim=-3).cuda()
        output = Variable(output, requires_grad=True)

        # Flatten and send to fully-connected linear layer
        output = torch.flatten(output)
        output = self.linear_layer(output)

        return output
