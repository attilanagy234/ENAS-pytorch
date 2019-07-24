import torch.nn as nn
import torch.nn.functional as F

class Controller(nn.Module):

   def __init__(self, num_layers, num_actions, lstm_size, lstm_num_layers, out_filters):
        super(Conroller, self).__init__()
        self.num_layers  = num_layers
        self.num_actions = num_actions
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.out_filters = out_filters


        lstm = nn.LSTM(input_size = lstm_size, hidden_size = lstm_size, num_layers=lstm_num_layers)

        g_embedding = nn.Embedding(num_embeddings = 1, embedding_size = lstm_size) #inputs = g_embedding().weight
        w_embedding = nn.Embedding(num_embeddings = num_actions, embedding_size = lstm_size)
        soft_embedding = nn.Embedding(num_embeddings = lstm_size, embedding_size = num_actions)

   def reset_params():
        raise NotImplementedError()

   def forward():

        architecture = dict()
        entropies = list()
        logprobs = list()
        inputs = g_embedding().weight #=learnable weight of the module of shape(num_embeddings, embedding_dim) initialized as N(0,1)
        hidden_state = None # =nulltensor


        for layer_i in range(self.num_layers):
            #forward prop
            inputs = inputs.unsqueeze(0)

            outputs, hidden_state_next = self.lstm(inputs, hidden_state)
            outputs.unsqeeze(0)

            hidden_state = hidden_state_next

            logits = soft_embedding(outputs)
            #apply temperature / tanh_const

            #sample an action
            action_distribution = Categorical(logits=logits)
            action_id = action_distribution.sample()

            architecture[str(layer_i)] = action_id

            logprobs.append(action_distribution.log_prob(action_id))

            entropys.append(action_distribution.entropy())

            inputs = w_embedding(action_id).unsqueeze(0)


        self.sampled_architecture = architecture
        self.sampled_entropies = entropies
        self.sampled_logprobs = logprobs





