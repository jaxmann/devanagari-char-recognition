import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################

        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(im_size[0], hidden_dim, kernel_size=kernel_size)
        self.height_width = (im_size[1] - (kernel_size) + 1)
        self.hidden_size = hidden_dim * self.height_width ** 2  # Channels x Height x Width

        self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        self.linear1 = nn.Linear(self.hidden_size / kernel_size ** 2, n_classes)

        self.kernel_size = kernel_size


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
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
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################

        x = F.relu(self.conv1(images))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, self.hidden_size / self.kernel_size ** 2)

        scores = self.linear1(x)


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

