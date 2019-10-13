import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.categorical import Categorical


class Controller(nn.Module):

    def __init__(self, writer, num_layers=2, num_branches=4, lstm_size=5, lstm_num_layers=2, tanh_constant=2.5,
                 temperature=5):

        super(Controller, self).__init__()

        self.writer = writer

        self.num_layers = num_layers
        self.num_branches = num_branches

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature

        self.w_lstm = nn.LSTM(input_size=self.lstm_size,
                              hidden_size=self.lstm_size,
                              num_layers=self.lstm_num_layers)

        # the decision in the previous step is fed as input embedding into the next step.
        # At the first step, the controller network receives an empty embedding as input.
        self.g_emb = nn.Embedding(1, self.lstm_size)  # Learn the  input of the lstm

        self.w_emb = nn.Embedding(self.num_branches, self.lstm_size)

        self.w_soft = nn.Linear(self.lstm_size, 2, bias=False)  # 2=choice for every parameter

        self.sampled_architecture = []
        self.sampled_entropies = []
        self.sampled_logprobs = []

        self._reset_params()

    def _reset_params(self):
        # print("reset params")

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

    def forward(self):
        h0 = None  # t0 hidden will be nullvec
        arc_seq = []
        entropies = []
        log_probs = []

        inputs = self.g_emb.weight

        for layer_id in range(self.num_layers):
            for branch_id in range(self.num_branches):

                inputs = inputs.view(1, 1, -1)
                output, hn = self.w_lstm(inputs, h0)
                output = output.squeeze(0)
                h0 = hn

                logit = self.w_soft(output)

                if self.temperature is not None:
                    logit /= self.temperature
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * torch.tanh(logit)

                out_dist = Categorical(logits=logit)
                out_id = out_dist.sample()
                arc_seq.append(out_id.item())

                log_prob = out_dist.log_prob(out_id)
                log_probs.append(log_prob.view(-1))

                entropy = out_dist.entropy()
                entropies.append(entropy.view(-1))

                inputs = self.w_emb(out_id)
                inputs = inputs.unsqueeze(0)

                self.writer.add_histogram("logits", logit)
                # self.writer.add_histogram("out_dist.logprob", out_dist.log_prob())

        self.sampled_architecture = arc_seq

        entropies = torch.cat(entropies)
        self.sampled_entropies = torch.sum(entropies)

        log_probs = torch.cat(log_probs)
        self.sampled_logprobs = torch.sum(log_probs)
