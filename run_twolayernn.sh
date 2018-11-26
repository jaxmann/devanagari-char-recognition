#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model twolayernn \
    --hidden-dim 10 \
    --epochs 3 \
    --weight-decay 0.9 \
    --momentum 0.8 \
    --batch-size 50 \
    --lr 0.0001 | tee twolayernn.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
