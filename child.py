import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor


class Child(nn.Module):

    def __init__(self, config, lr, momentum, num_classes, input_shape):
        super(Child, self).__init__()

        modules = nn.ModuleList()
        height_width = input_shape
        last_out_dim = 0

        for layer_param in config.values():

            modules.append(nn.Conv2d(in_channels=layer_param.input_dim,
                                           out_channels=layer_param.output_dim,
                                           kernel_size=layer_param.kernel_size,
                                           stride=layer_param.stride,
                                           dilation=1,
                                           groups=1,
                                           bias=True,
                                           padding=2)
                           )


            modules.append(torch.nn.ReLU())

            modules.append(torch.nn.MaxPool2d(kernel_size=layer_param.pooling_size,
                                                       stride=1,
                                                       dilation=1,
                                                       return_indices=False,
                                                       ceil_mode=False)
                           )

            height_width = self._conv2d_output_shape(height_width, layer_param.kernel_size, layer_param.stride, 2, 1)
            last_out_dim = layer_param.output_dim

            height_width = self._pooling2d_output_shape(height_width, kernel_size=layer_param.pooling_size, stride=1, dilation=1)

            #print(height_width, layer_param.output_dim)


        convOut_dim = height_width[0] *  height_width[0] * last_out_dim

        self.net = nn.Sequential(*modules)
        self.fc1 = nn.Linear(in_features=convOut_dim, out_features=500)
        self.fc2 = nn.Linear(500, num_classes)

        self.optimizer = torch.optim.SGD(self.net.parameters(),lr=lr, momentum=momentum)

    def _conv2d_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
        w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
        return h, w


    def _pooling2d_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        """(W1-f)/s +1 , H2=(H1-f)/s+1"""
        h = floor(((h_w[0] - kernel_size) // stride) + 1)
        w = floor(((h_w[1] - kernel_size) // stride) + 1)
        return h, w


    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
