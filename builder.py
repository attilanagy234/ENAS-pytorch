import torch
import torch.nn as nn
from collections import namedtuple


layer = namedtuple('kernel_size  padding pooling_size input_dim output_dim')


class Builder(object):

    def __init__(self, input_dim, output_dim, learning_rate, optimiser, param_per_layer, num_of_layers):
        # dropout, batch_norm could also be added
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.optimiser = optimiser
        self.param_per_layer = param_per_layer
        self.num_of_layers = num_of_layers

    def make_config(self, raw_config):
        config = dict()

        prev_dim = self.input_dim
        list_config = raw_config.split()

        for layer_i in range(self.num_of_layers, self.param_per_layer):

            kernel_size = 3 if list_config[layer_i*self.param_per_layer+0]<4 else 5
            padding = 3 if list_config[layer_i*self.param_per_layer+1]<4 else 5
            pooling_size = 3 if list_config[layer_i*self.param_per_layer+2]<4 else 5
            input_dim = prev_dim
            output_dim = round(list_config[layer_i*self.param_per_layer+3])
            prev_dim = output_dim

            current = layer(kernel_size, padding, pooling_size, input_dim, output_dim)

            config["layer_" + layer_i] = current

        return config

