import torch
import torch.nn as nn
import torch.nn.functional as F


class Child(nn.Module):

    def __init__(self, config, lr, l2_decay, num_classes):
        super(Child, self).__init__()

        list = nn.ModuleList()
        for item in config.values():
            current_conv_layer = nn.Conv2d(in_channels=item.input_dim,
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
            list.append(nn.modules.activation.LeakyReLU())
            list.append(current_pooling_layer)

        self.net = nn.Sequential(*list)

        self.fc1 = nn.Linear(20, 20)  # TODO : refactor
        self.fc2 = nn.Linear(20, num_classes)

        self.optimizer = torch.optim.SGD(self.net.parameters(),
                                         lr=lr, momentum=0.9,
                                         nesterov=True)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
