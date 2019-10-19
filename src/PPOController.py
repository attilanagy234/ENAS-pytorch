from collections import namedtuple

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import numpy as np


class Memory():

    #szepinteni lehetne, csak dic or csak list
    def __init__(self, num_layers):

        self.num_layers = num_layers
        self.transitions = []

        self.layer_ids = []
        self.inputs = []
        self.hiddenstates = []
        self.anchors = []
        self.anchors_w1 = []
        self.branches = []
        self.branch_logprobs = []
        self.skip_connections = []
        self.skip_logprobs = []
        self.rewards = []

    def add_transition(self, transition):

        self.transitions.append(transition)

        #skip_logprob: IS THE SUM OF ALL LOGPROBS, more layer is higher

    def add_transition2(self, transition):

        self.transitions.append(transition)

        self.layer_ids.append(transition["layer_id"])
        self.inputs.append(transition["inputs"])
        self.hiddenstates.append(transition["h0"])
        self.anchors.append(transition["anchors"])
        self.anchors_w1.append(transition["anchors_w1"])
        self.branches.append(transition["branch"])
        self.branch_logprobs.append(transition["branch_logporb"])
        self.skip_connections.append(transition["skip_connections"])
        self.skip_logprobs.append(transition["skip_logprob"])

    def add_rewards(self, reward):
        for _ in range(self.num_layers):
            self.rewards.append(reward)

    def clean(self):
        del self.transitions[:]
        del self.rewards[:]
        del self.layer_ids[:]
        del self.inputs[:]
        del self.hiddenstates[:]
        del self.anchors[:]
        del self.anchors_w1[:]
        del self.branches[:]
        del self.branch_logprobs[:]
        del self.skip_connections[:]
        del self.skip_logprobs[:]

    def get_logprobs(self):
        #branch_logprobs = []
        #skip_logprobs = []

        #for tx in self.transitions:
        #    branch_logprobs.append(tx["branch_logporb"])
        #    skip_logprobs.append(tx["skip_logprob"])


        skip_logprobs = np.array(self.skip_logprobs, dtype=float)  # you will have np.nan from None
        np.nan_to_num(skip_logprobs, copy=False)

        return self.branch_logprobs, skip_logprobs


class PPOController(nn.Module):

    def __init__(self, writer, num_layers=4, num_branches=6, lstm_size=5, lstm_num_layers=2, tanh_constant=2.5,
                 temperature=5, skip_target=0.8):


        super(PPOController, self).__init__()

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

        self.memory = Memory(self.num_layers)

    def _reset_params(self):
        #print("reset params")
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)
        #nn.init.uniform_(self.w_lstm.all_weights, -0.1, 0.1)

    def forward(self):
        h0 = None  # t0 hidden will be nullvec

        arc_seq = dict()
        branch_log_probs = []
        branch_entropies = []


        anchors = []
        anchors_w1 = []

        skip_counts = []
        skip_penalties = []
        skip_log_probs = []
        skip_entropies = []

        inputs = self.g_emb.weight

        for layer_id in range(self.num_layers):

            inputs = inputs.view(1, 1, -1)
            transition = dict()
            transition["layer_id"]=layer_id                             # layerid
            transition["inputs"]=inputs                               # inputs
            transition["h0"]=h0                                   # h0
            transition["anchors"] = anchors.copy()      # anchors
            transition["anchors_w1"] = anchors_w1.copy()  # anchorw


            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0)
            h0 = hn

            logit = self.w_soft(output)
#
            if self.temperature is not None:
                logit /= self.temperature

            out_dist = Categorical(logits=logit)
            out_id = out_dist.sample()

            arc_seq[str(layer_id)] = [(out_id.item())]

            transition["branch"]=out_id                                    # branch

            log_prob = out_dist.log_prob(out_id)
            branch_log_probs.append(log_prob.view(-1))

            transition["branch_logporb"]=log_prob.view(-1)                   # branch_logprob

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
                transition["skip_connections"]=skip_connections                     #skipconnection

                # get the kl_divergence from skip_target
                skip_prob = torch.sigmoid(logits)
                kl_div = skip_prob * torch.log(skip_prob / self.skip_target)
                kl_div = torch.sum(kl_div)
                skip_penalties.append(kl_div)

                # get the log probability
                skip_log_prob = skip_distribution.log_prob(skip_connections)
                skip_log_prob = torch.sum(skip_log_prob)
                skip_log_probs.append(skip_log_prob.view(-1))

                transition["skip_logprob"]=skip_log_prob                         #skip_log_prob

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
                transition["skip_connections"] = []
                transition["skip_logprob"] = None


            anchors.append(output)
            anchors_w1.append(self.w_attn_2(output))

            self.memory.add_transition2(
               transition
            )

        self.sampled_architecture = arc_seq

        branch_entropies = torch.cat(branch_entropies)
        skip_entropies = torch.cat(skip_entropies)
        self.sampled_entropies = torch.sum(torch.cat([branch_entropies, skip_entropies]))

        branch_log_probs = torch.cat(branch_log_probs)
        skip_log_probs = torch.cat(skip_log_probs)
        self.sampled_logprobs = torch.sum(torch.cat([branch_log_probs, skip_log_probs]))

    def evaluate(self, transition):

        output, hn = self.w_lstm(transition["inputs"], transition["h0"])
        output = output.squeeze(0)
        h0 = hn

        logits = self.w_soft(output)
#
        if self.temperature is not None:
            logits /= self.temperature

        out_dist = Categorical(logits=logits)

        branch_logprob = out_dist.log_prob(transition["branch"])

        branch_entropy = out_dist.entropy()

        inputs = self.w_emb(transition["branch"])
        inputs = inputs.unsqueeze(0)

        output, hn = self.w_lstm(inputs, h0)
        output = output.squeeze(0)

        if transition["layer_id"] > 0:

            query = torch.cat(transition["anchors_w1"], dim=0)

            query = torch.tanh(query+self.w_attn_2(output))
            query = self.v_attn(query)
            logits = torch.cat([query, -query], dim=1)

            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                logits = self.tanh_constant * torch.tanh(logits)

            #print("logits_eval", logits)
            skip_distribution = Categorical(logits=logits)

            skip_logprob = skip_distribution.log_prob(transition["skip_connections"])
            skip_logprob = torch.sum(skip_logprob)  # maybe without sum()

            skip_entropy = skip_distribution.entropy()
            skip_entropy = torch.sum(skip_entropy)

        else:
            skip_logprob = 0
            skip_entropy = 0

        return branch_logprob[0], skip_logprob, branch_entropy[0], skip_entropy


#
# DONE PPO
# -update weights
#   -old policy
# -memory
# -eval action in controller