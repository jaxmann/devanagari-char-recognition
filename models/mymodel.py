import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################

        # this is pretty much VGG11
        self.model = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            #changed final average pooling size for simpler linear layer
            nn.AvgPool2d(kernel_size=2, stride=2)

        )

        self.linear = nn.Linear(512, n_classes)

        # self.n_classes = n_classes
        #
        # self.conv1 = nn.Conv2d(im_size[0], 64, kernel_size=kernel_size)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.linear1 = nn.Linear(64*23*23, n_classes)
        #
        # self.pool1 = nn.MaxPool2d(4)
        # self.pool2 = nn.MaxPool2d(3)
        # # self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size)
        # # self.bn1 = nn.BatchNorm2d(hidden_dim)
        # # self.bn2 = nn.BatchNorm2d(64)
        #
        # # self.height_width = (12 - (kernel_size -1) -1 + 1)
        # self.hidden_size = 64 * (2) ** 2 # Channels x Height x Width
        #
        #
        # self.h2 = (14 - (kernel_size -1) -1 + 1)
        #
        # # self.linear1 = nn.Linear(self.hidden_size, 1000)
        # # self.linear2 = nn.Linear(1000, 100)
        # # self.linear3 = nn.Linear(100, n_classes)
        #
        #
        # self.kernel_size = kernel_size




        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
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
        # TODO: Implement the forward pass.
        #############################################################################

        # x = self.pool1(images)

        out = self.model(images)
        # print(out.shape)
        out = out.view(-1, 512)
        # out = out.view(out.size(0), -1)
        scores = self.linear(out)
        return scores

        # x = self.conv1(images)
        # x = F.relu(self.bn1(x))
        # # print(x.shape)
        # x = F.avg_pool2d(x, 8, 1)
        # # print(x.shape)
        # x = x.view(-1, 64*23*23)
        # scores = self.linear1(x)
        # print(scores.shape)

        # x = self.conv2(x)
        # x = F.relu(self.bn2(x))
        # x = self.pool1(x)
        # x = self.pool2(x)
        # # print(x.shape)
        #
        # # x = F.relu(self.conv2(x))
        # x = x.view(-1,self.hidden_size)
        # #
        # x = self.linear1(x)
        # x = self.linear2(x)
        # scores = (self.linear3(x))



        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

