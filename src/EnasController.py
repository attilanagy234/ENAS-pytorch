import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.categorical import Categorical

class EnasController(nn.Module):

    def __init__(self, writer, num_layers=4, num_branches=6, lstm_size=5, lstm_num_layers=2, tanh_constant=1.5,
                 temperature=None, skip_target = 0.8):


        super(EnasController, self).__init__()

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

        self.g_emb = nn.Embedding(1, self.lstm_size)  # Learn the  input of the lstm

        self.w_emb = nn.Embedding(self.num_branches, self.lstm_size)

        self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=False)  # 2=choice for every parameter

        # skip connections:
        self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)

        self.sampled_architecture = []
        self.sampled_entropies = []
        self.sampled_logprobs = []

        self.skip_target = skip_target

        self._reset_params()

    def _reset_params(self):
        #print("reset params")

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

    def forward(self):
        h0 = None  # t0 hidden will be nullvec

        arc_seq = dict()

        branch_entropies = []
        branch_log_probs = []

        anchors = []
        anchors_w1 = []

        skip_counts = []
        skip_penalties = []
        skip_log_probs = []
        skip_entropies = []

        inputs = self.g_emb.weight

        for layer_id in range(self.num_layers):
            inputs = inputs.view(1, 1, -1)
            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0)
            h0 = hn

            logit = self.w_soft(output)
#
            if self.temperature is not None:
                logit /= self.temperature

            out_dist = Categorical(logits=logit)
            out_id = out_dist.sample()

            arc_seq[str(layer_id)] = [(out_id.item())] #TODO: rewrite trainer:make_config

            log_prob = out_dist.log_prob(out_id)
            branch_log_probs.append(log_prob.view(-1))

            entropy = out_dist.entropy()
            branch_entropies.append(entropy.view(-1))

            inputs = self.w_emb(out_id)
            inputs = inputs.unsqueeze(0)

            # self.writer.add_histogram("logits", logit)
            # self.writer.add_histogram("out_dist.logprob", out_dist.log_prob())

            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0)

            if layer_id > 0:

                # propagate the lstm output through the linear layers
                query = torch.cat(anchors_w1, dim=0)
                query = torch.tanh(query+self.w_attn_2(output))
                query = self.v_attn(query)
                logits = torch.cat([query, -query], dim=1)

                if self.temperature is not None:
                    logits /= self.temperature
                if self.tanh_constant is not None:
                    logits = self.tanh_constant * torch.tanh(logits)

                # sample skip connections from the output
                skip_distribution = Categorical(logits=logits)
                #print(logits)
                skip_connections = skip_distribution.sample().view(layer_id)
                arc_seq[str(layer_id)].append(skip_connections)

                # get the kl_divergence from skip_target
                skip_prob = torch.sigmoid(logits)
                kl_div = skip_prob * torch.log(skip_prob / self.skip_target)
                kl_div = torch.sum(kl_div)
                skip_penalties.append(kl_div)

                # get the log probability
                skip_log_prob = skip_distribution.log_prob(skip_connections)
                skip_log_prob = torch.sum(skip_log_prob)
                skip_log_probs.append(skip_log_prob.view(-1))

                # get entropy
                skip_entropy = skip_distribution.entropy()
                skip_entropy = torch.sum(skip_entropy)
                skip_entropies.append(skip_entropy.view(-1))
                # print(skip_entropies)

                # get the number of skipcounts
                skip_count = torch.sum(skip_connections)
                skip_counts.append(skip_count)


                # calculate the next input of the lstm
                inputs = torch.matmul(skip_connections.float(), torch.cat(anchors, dim=0))

                inputs /= (1.0 + skip_count)
            else:
                arc_seq[str(layer_id)].append([])
                inputs = self.g_emb.weight

            anchors.append(output)
            anchors_w1.append(self.w_attn_2(output))

        self.sampled_architecture = arc_seq

        branch_entropies = torch.cat(branch_entropies)
        # print(skip_entropies)
        skip_entropies = torch.cat(skip_entropies)
        self.sampled_entropies = torch.sum(torch.cat([branch_entropies, skip_entropies]))

        branch_log_probs = torch.cat(branch_log_probs)
        skip_log_probs = torch.cat(skip_log_probs)
        self.sampled_logprobs = torch.sum(torch.cat([branch_log_probs, skip_log_probs]))
