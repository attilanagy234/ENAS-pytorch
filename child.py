import torch
import torch.nn as nn


class Child(nn.Module):

    def __init__(self, config, lr, l2_decay):
        super(Child, self).__init__()

        list = nn.ModuleList()
        for item in config.values():

            current_conv_layer = nn.conv2D(in_channels=item.input_dim,
                                           out_channels=item.output_dim,
                                           kernel_size=item.kernel_size,
                                           stride=1,
                                           dilation=1,
                                           groups=1,
                                           bias=True,
                                           padding_mode='zeros')

            current_pooling_layer = torch.nn.MaxPool2d(kernel_size=item.kernel_size,
                                                       stride=None,
                                                       dilation=1,
                                                       return_indices=False,
                                                       ceil_mode=False)
            list.append(current_conv_layer)
            list.append(nn.ReLU)
            list.append(current_pooling_layer)

        list.append(torch.flatten)
        list.append(nn.Linear)
        list.append(nn.Linear)
        self.net = nn.Sequential(list)

        self.optimizer = torch.optim.SGD(self.net.parameters(),
                                         lr=lr, momentum=0.9,
                                         nesterov=True,
                                         weight_decay=l2_decay)

    def forward(self, x):
        x = self.net(x)
        return x
