import torch

from controller import Controller
from builder import Builder
import torch.nn.functional as F

from collections import namedtuple

layer = namedtuple('layer', 'kernel_size  padding pooling_size input_dim output_dim')


class NetManager(object):

    def __init__(self, log_interval, num_of_children, input_dim, output_dim, lr, param_per_layer, num_of_layers):

        self.log_interval = log_interval
        self.num_of_children = num_of_children

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.param_per_layer = param_per_layer
        self.num_of_layers = num_of_layers

        self.controller = Controller()
        self.children = list()


    def make_config(self, raw_config):
        config = dict()

        prev_dim = self.input_dim
        list_config = raw_config.split()

        for layer_i in range(self.num_of_layers, self.param_per_layer):
            kernel_size = 3 if list_config[layer_i * self.param_per_layer + 0] < 4 else 5
            padding = 3 if list_config[layer_i * self.param_per_layer + 1] < 4 else 5
            pooling_size = 3 if list_config[layer_i * self.param_per_layer + 2] < 4 else 5
            input_dim = prev_dim
            output_dim = round(list_config[layer_i * self.param_per_layer + 3])
            prev_dim = output_dim

            current = layer(kernel_size, padding, pooling_size, input_dim, output_dim)

            config["layer_" + layer_i] = current

        return config

    def sample_child(self):
        return self.controller.sample()

    def sample_children(self):
        for i in range(self.num_of_children):
            self.children.append(self.controller.sample())

    def train_controller(self, model, device, data_loader, optimizer, epoch, entropy_weight=0.0001):
        model.train()

        valid_loader = data_loader["validation"]
        train_loader = data_loader["train"]

        for epoch_idx in range(epoch):
            loss = torch.FloatTensor([0])
            for child_idx in range(self.num_of_children):
                # images, labels.cuda()

                model()  # forward pass without input
                sampled_architecture = model.sampled_architecture
                sampled_entropies = model.sampled_entropies
                sampled_logprobs = model.sampled_logprobs

                # get the acc of a single child

                child = self.builder(sampled_architecture)
                child.train(train_loader)
                validation_accuracy = child.test(valid_loader)

                # with torch.no_grad():
                #    prediction = builder(sampled_architecture)(images)
                # validation_accuracy = torch.mean((torch.max(prediction, 1)[1] == labels).type(torch.float)) #TODO: this shoudl be changed according to the data
                # or F.nll_loss(prediction, labels, reduction = "sum").item()  #sum up batch loss

                reward = torch.tensor(100 - validation_accuracy).detach()
                reward += sampled_entropies * entropy_weight

                # calculate advantage with baseline (moving avg)

                loss += sampled_logprobs * reward

            loss /= self.num_of_children
            loss.backwards(retrain_graph=True)  # retrain_graph: keep the gradients, idk if we need this but tdvries does

            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), child_grad_bound) #normalize gradient
            optimizer.step()
            model.zero_grad()

    def train_child(self, child, device, train_loader):

        child.train()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            child.optimizer.zero_grad()
            prediction = child(images)
            loss = F.nll_loss(prediction, labels)
            loss.backwards()
            child.optimizer.step()

            print(loss)

    def test_child(self, child, device, valid_loader):

        child.eval()
        validation_loss = 0
        correct = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                prediction_probs = child(images)
                validation_loss += F.nll_loss(prediction_probs, labels, reduction="sum").item()  # batch loss
                prediction = prediction_probs.arg_max(dim=1, keepdim=True)  # index of max logprob
                correct += prediction.eq(labels.view_as(prediction)).sum().item()

        validation_loss /= len(valid_loader.dataset)
