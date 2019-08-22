import torch
import torch.nn as nn
import torch.nn.functional as F
from EnasLayer import *
from math import floor
import tensorflow as tf


class EnasChild(nn.Module):

    def __init__(self, enas_config, num_layers, lr=0.01, momentum=0.5, num_classes=10, num_channels=1, input_shape=28, out_filters=10, num_branches=6, fixed_arc=True ):
        super(EnasChild, self).__init__()
        self.config = enas_config
        self.num_branches = num_branches
        self.fixed_arc = fixed_arc
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.out_filters = out_filters
        self.num_layers = num_layers
        self.layerList = nn.ModuleList([])
        self.reductionList = nn.ModuleList([]) # for reduction layers


        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

        self.layerList.append(nn.Conv2d(num_channels, self.out_filters, kernel_size=3, padding=1, bias=False))
        self.layerList.append(nn.BatchNorm2d(out_filters, track_running_stats=False))

        for layer_id, layer_param in self.config.items():
            currentLayer = FixedEnasLayer(in_filters=out_filters, out_filters=out_filters, layer_type=layer_param, layer_id=layer_id, prev_layers="unused")
            self.layerList.append(currentLayer)
            last_kernel = currentLayer.layer.kernel

            if len(self.layerList) in self.pool_layers:
                if fixed_arc:
                    #add factorized reduction
                    #double the number of outfilters
                    self.pool_layers.append()
                if not fixed_arc:
                    #add factorizedReduction
                    #why no doubleout_filter?
                    raise NotImplementedError("enasChild: fixed_arc=true")

        self.net = nn.Sequential(*self.layerList)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=out_filters, out_features=50)
        self.fc2 = nn.Linear(50, num_classes)

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)

    def forward(self, x):

        x = self.net(x)
        x=self.global_avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
