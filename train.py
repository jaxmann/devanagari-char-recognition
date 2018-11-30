# NOTE: The scaffolding code for this part of the assignment
# is adapted from https://github.com/pytorch/examples.
from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torchvision
from torchvision.models import alexnet
from torchvision.models import vgg16
import torchvision.models
from torch.optim import lr_scheduler
import time
import copy



from cifar10 import CIFAR10

# You should implement these (softmax.py, twolayernn.py, convnet.py)
import models.softmax 
import models.twolayernn
import models.convnet
import models.mymodel
import models.SimpleNet
import scipy

import torchvision.models as m

# Training settings
parser = argparse.ArgumentParser(description='CIFAR-10 Example')
# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, metavar='M',
                    help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--model',
                    choices=['softmax', 'convnet', 'twolayernn', 'mymodel', 'SimpleNet'],
                    help='which model to train/evaluate')
parser.add_argument('--hidden-dim', type=int,
                    help='number of hidden features/activations')
parser.add_argument('--kernel-size', type=int,
                    help='size of convolution kernels/filters')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='number of batches between logging train status')
parser.add_argument('--cifar10-dir', default='data',
                    help='directory that contains cifar-10-batches-py/ '
                         '(downloaded automatically if necessary)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load CIFAR10 using torch data paradigm
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# CIFAR10 meta data
n_classes = 46
im_size = (3, 224, 224)
# Subtract the mean color and divide by standard deviation. The mean image
# from part 1 of this homework was essentially a big gray blog, so
# subtracting the same color for all pixels doesn't make much difference.
# mean color of training images
# cifar10_mean_color = [0.49131522, 0.48209435, 0.44646862]
# # std dev of color across training images
# cifar10_std_color = [0.01897398, 0.03039277, 0.03872553]
# transform = transforms.Compose([
#                  transforms.ToTensor(),
#                  transforms.Normalize(cifar10_mean_color, cifar10_std_color),
#             ])

def load_dataset(data_path):
    tforms = [torchvision.transforms.Resize(size=(224, 224)), torchvision.transforms.ToTensor()]
    tf = transforms.Compose(tforms)
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=tf
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True
    )
    return train_loader

train_loader = load_dataset('data/train/')
val_loader = load_dataset('data/train')
test_loader =load_dataset('data/test')


dataloaders = {'train': train_loader,
               'val': val_loader}

dataset_sizes = {'train': len(train_loader),
               'val': len(val_loader)}

# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
#                                              shuffle=True, num_workers=4)
#               for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# Datasets
# train_dataset = CIFAR10(args.cifar10_dir, split='train', download=True,
#                         transform=transform)
# val_dataset = CIFAR10(args.cifar10_dir, split='val', download=True,
#                         transform=transform)
# test_dataset = CIFAR10(args.cifar10_dir, split='test', download=True,
#                         transform=transform)


# DataLoaders
# train_loader = torch.utils.data.DataLoader(train_dataset,
#                  batch_size=args.batch_size, shuffle=True, **kwargs)
# val_loader = torch.utils.data.DataLoader(val_dataset,
#                  batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset,
#                  batch_size=args.batch_size, shuffle=True, **kwargs)

# Load the model
if args.model == 'softmax':
    model = models.softmax.Softmax(im_size, n_classes)
elif args.model == 'twolayernn':
    model = models.twolayernn.TwoLayerNN(im_size, args.hidden_dim, n_classes)
elif args.model == 'convnet':
    model = models.convnet.CNN(im_size, args.hidden_dim, args.kernel_size,
                               n_classes)
elif args.model == 'mymodel':
    model = models.mymodel.MyModel(im_size, args.hidden_dim, args.kernel_size, n_classes)
elif args.model == 'SimpleNet':
    # model = models.SimpleNet.SimpleNet(n_classes, droprate=0.5, rgb=True)
    model = models.SimpleNet.create_part2_model(m.alexnet(pretrained=True), n_classes)
    #model = m.resnet18(pretrained=True)
    #num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, n_classes)
else:
    raise Exception('Unknown model {}'.format(args.model))
# cross-entropy loss function
criterion = F.cross_entropy
if args.cuda:
    model.cuda()
    # criterion.cuda()

#############################################################################
# TODO: Initialize an optimizer from the torch.optim package using the
# appropriate hyperparameters found in args. This only requires one line.
#############################################################################

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# criterion = nn.CrossEntropyLoss()
#
# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #, weight_decay=args.weight_decay, momentum=args.momentum)
print(model)
if args.model == 'SimpleNet':

    print(model)
    params_to_optimize = []
    backprop_depth = 3
    # List of modules in the network
    mods = list(model.features.children()) + list(model.classifier.children())

    # Extract parameters from the last `backprop_depth` modules in the network and collect them in
    # the params_to_optimize list.
    for m in mods[::-1][:backprop_depth]:
        params_to_optimize.extend(list(m.parameters()))

    optimizer = torch.optim.Adam(params=params_to_optimize, lr=args.lr) #, weight_decay=args.weight_decay, momentum=args.momentum)


#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

def train(epoch):
    '''
    Train the model for one epoch.
    '''
    # Some models use slightly different forward passes and train and test
    # time (e.g., any model with Dropout). This puts the model in train mode
    # (as opposed to eval mode) so it knows which one to use.
    model.train()
    # train loop
    for batch_idx, batch in enumerate(train_loader):
        # prepare data
        images, targets = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            images, targets = images.cuda(), targets.cuda()
        #############################################################################
        # TODO: Update the parameters in model using the optimizer from above.
        # This only requires a couple lines of code.
        #############################################################################

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        if batch_idx % args.log_interval == 0:
            val_loss, val_acc = evaluate('val', n_batches=4)
            train_loss = loss.data[0]
            examples_this_epoch = batch_idx * len(images)
            epoch_progress = 100. * batch_idx / len(train_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
                epoch, examples_this_epoch, len(train_loader.dataset),
                epoch_progress, train_loss, val_loss, val_acc))

def evaluate(split, verbose=False, n_batches=None):
    '''
    Compute loss on val or test data.
    '''
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    if split == 'val':
        loader = val_loader
    elif split == 'test':
        loader = test_loader
    for batch_i, batch in enumerate(loader):
        data, target = batch
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss += criterion(output, target, size_average=False).data[0]
        # predict the argmax of the log-probabilities
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        n_examples += pred.size(0)
        if n_batches and (batch_i >= n_batches):
            break

    loss /= n_examples
    acc = 100. * correct / n_examples
    if verbose:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            split, loss, correct, n_examples, acc))
    return loss, acc


# train the model one epoch at a time
for epoch in range(1, args.epochs + 1):
    train(epoch)
evaluate('test', verbose=True)

# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 scheduler.step()
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
#
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]
#
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))
#
#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#
#         print()
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model
#
# model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)

# Save the model (architecture and weights)
torch.save(model, args.model + '.pt')
# Later you can call torch.load(file) to re-load the trained model into python
# See http://pytorch.org/docs/master/notes/serialization.html for more details


