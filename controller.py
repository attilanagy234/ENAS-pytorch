import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Controller(nn.Module):

    def __init__(self, num_layers, num_actions, lstm_size, lstm_num_layers, out_filters):
        super(Controller, self).__init__()
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.out_filters = out_filters

        self.lstm = nn.LSTM(input_size=lstm_size, hidden_size=lstm_size, num_layers=lstm_num_layers)

        self.g_embedding = nn.Embedding(num_embeddings=1, embedding_dim=lstm_size)  # inputs = g_embedding().weight
        self.w_embedding = nn.Embedding(num_embeddings=num_actions, embedding_dim=lstm_size)
        self.soft_embedding = nn.Embedding(num_embeddings=lstm_size, embedding_dim=num_actions)

        self.sampled_architecture = list()
        self.sampled_entropies = list()
        self.sampled_logprobs = list()

    def reset_params(self):
        raise NotImplementedError()

    def forward(self):
        architecture = dict()
        entropies = list()
        logprobs = list()
        inputs = self.g_embedding().weight  # =learnable weight of the module of shape(num_embeddings, embedding_dim) initialized as N(0,1)
        hidden_state = None  # =nulltensor

        for layer_i in range(self.num_layers):
            # forward prop
            inputs = inputs.unsqueeze(0)

            outputs, hidden_state_next = self.lstm(inputs, hidden_state)
            outputs.unsqeeze(0)

            hidden_state = hidden_state_next

            logits = self.soft_embedding(outputs)
            # apply temperature / tanh_const

            # sample an action
            action_distribution = Categorical(logits=logits)
            action_id = action_distribution.sample()

            architecture[str(layer_i)] = action_id

            logprobs.append(action_distribution.log_prob(action_id))

            entropies.append(action_distribution.entropy())

            inputs = self.w_embedding(action_id).unsqueeze(0)

        self.sampled_architecture = architecture
        self.sampled_entropies = entropies
        self.sampled_logprobs = logprobs
