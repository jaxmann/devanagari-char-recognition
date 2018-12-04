import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from sklearn.preprocessing import StandardScaler
import os.path as osp
# import utils
import torch.optim as optim



# class SimpleNet(nn.Module):
#     """
#     This class implements the network model needed for part 1. Network models in
#     pyTorch are inherited from torch.nn.Module, only require implementing the
#     __init__() and forward() methods. The backpropagation is handled automatically
#     by pyTorch.
#
#     The __init__() function defines the various operators needed for
#     the forward pass e.g. conv, batch norm, fully connected, etc.
#
#     The forward() defines how these blocks act on the input data to produce the
#     network output. For hints on how to implement your network model, see the
#     AlexNet example at
#     https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
#     """
#
#     def __init__(self, num_classes, droprate=0.5, rgb=False, verbose=False):
#         """
#         This is where you set up and initialize your network. A basic network is
#         already set up for you. You will need to add a few more layers to it as
#         described. You can refer to https://pytorch.org/docs/stable/nn.html for
#         documentation.
#
#         Args:
#         - num_classes: (int) Number of output classes.
#         - droprate: (float) Droprate of the network (used for droppout).
#         - rgb: (boolean) Flag indicating if input images are RGB or grayscale, used
#           to set the number of input channels.
#         - verbose: (boolean) If True a hook is registered to print the size of input
#           to classifier everytime the forward function is called.
#         """
#         super(SimpleNet, self).__init__()  # initialize the parent class, a must
#         in_channels = 3 if rgb else 1
#
#         """ NETWORK SETUP """
#         #####################################################################
#         #                       TODO: YOUR CODE HERE                        #
#         #####################################################################
#         # TODO modify the simple network
#         # 1) add one dropout layer after the last relu layer.
#         # 2) add more convolution, maxpool and relu layers.
#         # 3) add one batch normalization layer after each convolution/linear layer
#         #    except the last convolution/linear layer of the WHOLE model (meaning
#         #    including the classifier).
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=9,
#                       stride=1, padding=0, bias=False),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(10),
#             nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5,
#                       stride=1, padding=0, bias=False),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(10),
#             nn.Dropout(p=0.8)
#         )
#
#
#
#         self.a = nn.Dropout(p=0.8)
#
#         self.classifier = nn.Conv2d(in_channels=10, out_channels=num_classes,
#                                     kernel_size=11, stride=1, padding=0)
#
#
#
#         """ NETWORK INITIALIZATION """
#         for name, m in self.named_modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 # Initializing weights with randomly sampled numbers from a normal
#                 # distribution.
#                 m.weight.data.normal_(0, 1)
#                 m.weight.data.mul_(1e-2)
#                 if m.bias is not None:
#                     # Initializing biases with zeros.
#                     nn.init.constant_(m.bias.data, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#
#                 # m.weight.data.normal_(0, 1)
#                 # m.weight.data.mul_(1e-2)
#                 nn.init.constant_(m.weight.data, 1)
#                 if m.bias is not None:
#                     # Initializing biases with zeros.
#                     nn.init.constant_(m.bias.data, 0)
#
#
#
#         if verbose:
#             # Hook that prints the size of input to classifier everytime the forward
#             # function is called.
#             self.classifier.register_forward_hook(utils.print_input_size_hook)
#
#     def forward(self, x):
#         """
#         Forward step of the network.
#
#         Args:
#         - x: input data.
#
#         Returns:
#         - x: output of the classifier.
#         """
#         # first layer
#         x = self.features(x)
#
#         x = self.classifier(x)
#
#         return x.squeeze()



def create_model(model, num_classes):
    """
    Take the passed in model and prepare it for finetuning by following the
    instructions.

    Args:
    - model: The model you need to prepare for finetuning. For the purposes of
      this project, you will pass in AlexNet.
    - num_classes: number of classes the model should output.

    Returns:
    - model: The model ready to be fine tuned.
    """
    # # Getting all layers from the input model's classifier.
    new_classifier = list(model.classifier.children())
    new_classifier = new_classifier[:-1]

    fc = nn.Linear(4096, num_classes)

    fc.weight.data.normal_(0, 1)
    fc.weight.data.mul_(1e-2)
    if fc.bias is not None:
        # Initializing biases with zeros.
        nn.init.constant_(fc.bias.data, 0)
    
    new_classifier[-1] = fc

    fc_prev = nn.Linear(4096, 4096)

    fc_prev.weight.data.normal_(0, 1)
    fc_prev.weight.data.mul_(1e-2)
    if fc_prev.bias is not None:
        # Initializing biases with zeros.
        nn.init.constant_(fc_prev.bias.data, 0)

    new_classifier[-2] = fc_prev

    # Connecting all layers to form a new classifier.
    model.classifier = nn.Sequential(*new_classifier)

    return model

