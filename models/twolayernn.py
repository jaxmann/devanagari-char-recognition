import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TwoLayerNN(nn.Module):
    def __init__(self, im_size, hidden_dim, n_classes):
        '''
        Create components of a two layer neural net classifier (often
        referred to as an MLP) and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            n_classes (int): Number of classes to score
        '''
        super(TwoLayerNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(im_size[0] * im_size[1] * im_size[2], hidden_dim), #hidden
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, images):
        '''
        Take a batch of images and run them through the NN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None

        N = images.size()[0]

        vect = images.view(N, -1)
        scores = self.model(vect)

        return scores

