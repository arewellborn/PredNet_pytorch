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
        self,
        prednet,
    ):
        super(PredNetDNI, self).__init__()

        # Freeze all of the existing PredNet layers
        for param in self.prednet.parameters():
            param.requires_grad = False

        # Spatially average pool lower layers to match upper layer dims
        self.pool_list = nn.ModuleList()
        for n in reversed(range(1, 4)):
            self.pool_list.append(nn.AvgPool2D(kernel_size=2 ** n, stride=2 ** n))
        
        # We are concatenating and flattening all elements for the out gate for all layers
        in_features = 8 * (3 + 48 + 96 + 192) * 30 * 30 # batch * all layer features * (h / 2 ^ 3) * (w / 2 ^ 3)
        self.linear_layer = Variable(nn.Linear(in_features=in_features, out_features=1), requires_grad=True)

    def forward(self, A0_withTimeStep, initial_states):

        # 默认是batch_fist == True的, 即第一维是batch_size, 第二维是timesteps.
        A0_withTimeStep = A0_withTimeStep.transpose(
            0, 1
        )  # (b, t, c, h, w) -> (t, b, c, h, w)

        num_timesteps = A0_withTimeStep.size()[0]

        hidden_states = initial_states  # 赋值为hidden_states是为了在下面的循环中可以无痛使用
        # No need to retain the prednet predictions since we only want the DNI prediction
        for t in range(num_timesteps):
            A0 = A0_withTimeStep[t, ...]
            prednet_output, hidden_states = self.prednet(A0, hidden_states)

        # Concat out gate layers across channel/feature axis after pooling
        output = []
        for i, layer in enumerate(self.prednet.conv_layers["o"]):
            if i == 0:
                output.append(layer)
            else:
                pool = self.pool_list[i - 1]
                output.append(pool(layer))
        output = torch.cat(output, dim=-3)

        # Flatten and send to fully-connected linear layer
        output = torch.flatten(output)
        output = self.linear_layer(output)

        return output
