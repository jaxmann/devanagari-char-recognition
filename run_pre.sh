#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality,
#       number of epochs, weigh decay factor, momentum, batch size, learning
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model SimpleNet \
    --kernel-size 3 \
    --hidden-dim 50 \
    --epochs 6 \
    --weight-decay 0.0005 \
    --momentum 0.9 \
    --batch-size 512 \
    --lr 0.001 | tee alexTransfer.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
