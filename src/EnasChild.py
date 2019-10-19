import torch
import torch.nn as nn
import torch.nn.functional as F
from EnasLayer import *
from math import floor


class FixedEnasChild(nn.Module):

    def __init__(self, enas_config, num_layers, lr,input_shape, out_filters, keep_prob=0.8, momentum=0.5, num_classes=10,  num_branches=6, input_channels=3,  t0=10, eta_min=0.001):
        super(FixedEnasChild, self).__init__()
        self.config = enas_config
        self.num_branches = num_branches
        self.keep_prob = keep_prob
        self.input_channels = input_channels
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.out_filters = out_filters
        self.num_layers = num_layers
        self.layerList = nn.ModuleList([])
        self.reductionList = nn.ModuleList([])  # for reduction layers


        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

        self.layerList.append(nn.Conv2d(input_channels, self.out_filters, kernel_size=3, padding=1, bias=False))
        self.layerList.append(nn.BatchNorm2d(out_filters, track_running_stats=False))

        for layer_id, layer_param in self.config.items():

            currentLayer = FixedEnasLayer(in_filters=out_filters, out_filters=out_filters, branch_id=layer_param[0], layer_id=int(layer_id), prev_layers="unused")

            self.layerList.append(currentLayer)
            last_kernel = currentLayer.layer.kernel


            # bug: expected 10 channels but got 20 instead:
            # if int(layer_id) in self.pool_layers:
            #
            #     reductionLayer = FactorizedReduction(out_filters, out_filters*2, 2)
            #     self.out_filters *= 2
            #     self.layerList.append(reductionLayer)

            if int(layer_id) in self.pool_layers:

                reductionLayer = FactorizedReduction(out_filters, out_filters, 2)
                self.layerList.append(reductionLayer)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=1. - self.keep_prob)
        self.fc1 = nn.Linear(in_features=out_filters, out_features=num_classes)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t0, eta_min=eta_min, last_epoch=-1) #according to enas paper: lmax = 0.05, lmin=0.001, t0 = 10, tmul=2


    def forward(self, x, config):

        prev_outputs = []

        current_enaslayer = 0

        for layer_idx, layer in enumerate(self.layerList):
            if isinstance(layer, FixedEnasLayer):
                # or isinstance(self.layerList[layer_idx], FixedEnasLayer):

                x = self.layerList[layer_idx](x, config[str(current_enaslayer)], prev_outputs)

                prev_outputs.append(x)
                current_enaslayer += 1

            # DOWNSAMPLE all the previuous outputs:
            elif isinstance(layer, FactorizedReduction):
                for out_idx, output in enumerate(prev_outputs):
                    x = self.layerList[layer_idx](prev_outputs[out_idx])
                    prev_outputs[out_idx] = x
                x = prev_outputs[-1]  # not needed
            else:
                x = self.layerList[layer_idx](x)

        x = self.global_avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        out = F.log_softmax(x, dim=1)

        return out

class SharedEnasChild(nn.Module):

    def __init__(self, num_layers, lr=0.01, keep_prob=0.8, momentum=0.5, num_classes=10,
                 input_channels=3, input_shape=32, out_filters=10, num_branches=6,
                 t0=10, eta_min=0.001, l2_decay=2e-4):

        super(SharedEnasChild, self).__init__()
        self.num_branches = num_branches
        self.keep_prob = keep_prob
        self.input_shape = input_shape
        self.num_channels = input_channels
        self.out_filters = out_filters
        self.num_layers = num_layers
        self.layerList = nn.ModuleList([])
        self.reductionList = nn.ModuleList([])  # for reduction layers
        self.l2_decay = l2_decay


        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

        self.stemConv = nn.Sequential(
            nn.Conv2d(self.num_channels, self.out_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_filters, track_running_stats=False)
        )

        for layer_id in range(self.num_layers):

            currentLayer = SharedEnasLayer(in_filters=out_filters, out_filters=out_filters, layer_id=layer_id+1,
                                           prev_layers="unused")

            self.layerList.append(currentLayer)
            self.layerList.append(nn.BatchNorm2d(out_filters, track_running_stats=False))

            if int(layer_id) in self.pool_layers:
                reductionLayer = FactorizedReduction(out_filters, out_filters, 2)
                self.layerList.append(reductionLayer)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=1. - self.keep_prob)
        self.fc1 = nn.Linear(in_features=out_filters, out_features=num_classes)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=self.l2_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t0, eta_min=eta_min, last_epoch=-1) #according to enas paper: lmax = 0.05, lmin=0.001, t0 = 10, tmul=2

    def forward(self, x, config):

        prev_outputs = []

        x = self.stemConv(x)
        current_enaslayer = 0

        for layer_idx, layer in enumerate(self.layerList):
            if isinstance(layer, SharedEnasLayer):
                # or isinstance(self.layerList[layer_idx], FixedEnasLayer):
                x = self.layerList[layer_idx](x, config[str(current_enaslayer)], prev_outputs)
                prev_outputs.append(x)
                current_enaslayer += 1

            # DOWNSAMPLE all the previuous outputs:
            elif isinstance(layer, FactorizedReduction):

                for out_idx, output in enumerate(prev_outputs):
                    x = self.layerList[layer_idx](prev_outputs[out_idx])
                    prev_outputs[out_idx] = x
                x = prev_outputs[-1] # not needed
            else:
                x = self.layerList[layer_idx](x)

        x = self.global_avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        out = F.log_softmax(x, dim=1)

        return out

