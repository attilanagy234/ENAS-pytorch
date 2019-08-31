from typing import Dict, Union
import torch
import torch.nn.functional as F
from collections import namedtuple
# from controller import Controller
from EnasController import EnasController
# from child import Child
from EnasChild import *
from utils import push_to_tensor_alternative, get_logger

layer = namedtuple('layer', 'kernel_size stride pooling_size input_dim output_dim')


class Trainer(object):

    def __init__(self,
                 writer,
                 log_interval,
                 num_of_children,
                 input_dim,
                 num_classes,
                 learning_rate_child,
                 momentum_child,
                 num_branches,
                 num_of_layers,
                 out_filters,
                 controller_size,
                 controller_layers,
                 isShared):

        self.writer = writer
        self.log_interval = log_interval
        self.num_of_children = num_of_children

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.out_filters = out_filters

        self.isShared = isShared

        self.num_branches = num_branches
        self.num_layers = num_of_layers

        self.learning_rate_child = learning_rate_child
        self.momentum = momentum_child

        self.controller = EnasController(self.writer,
                                         self.num_layers,
                                         self.num_branches,
                                         controller_size,
                                         controller_layers)  # self,num_layers=2,num_branches=4,lstm_size=5,lstm_num_layers=2,tanh_constant=1.5,temperature=None
        self.children = list()

        self.globaliter = 0
        self.logger = get_logger()

        if self.isShared:
            self.child = SharedEnasChild(self.num_layers, self.learning_rate_child, self.momentum,
                                         num_classes=self.num_classes, out_filters=self.out_filters,
                                         input_shape=self.input_dim)

    ## unused
    def make_config(self, raw_config):

        raise Exception("make_config: dont use this")

        config = dict()

        prev_dim = self.input_dim

        list_config = list(map(int, raw_config))

        for layer_i in range(0, self.num_of_layers * self.param_per_layer, self.param_per_layer):
            kernel_size = 3 if list_config[layer_i + 0] else 5
            stride = 1 if list_config[layer_i + 1] else 2
            pooling_size = 2 if list_config[layer_i + 2] else 2
            input_dim = prev_dim
            out_channels = 20 if list_config[layer_i + 3] else 30
            prev_dim = out_channels

            current = layer(kernel_size, stride, pooling_size, input_dim, out_channels)
            config[str(layer_i)] = current

        return config

    def make_enas_config(self, raw_config):

        config = dict()
        list_config = list(map(int, raw_config))

        for layer_i in range(0, self.num_layers):
            config[str(layer_i)] = list_config[layer_i]

        return config

    # unused
    def sample_child(self):
        return self.controller.sample()

    # unused
    def sample_children(self):
        for i in range(self.num_of_children):
            self.children.append(self.controller.sample())

    def train_controller(self, model, optimizer, device, train_loader, valid_loader, epoch, momentum,
                         entropy_weight):

        model.train()

        step = 0

        prev_runs = torch.zeros([5])  # to store the val_acc of prev epochs

        for epoch_idx in range(epoch):
            loss = torch.FloatTensor([0])
            epoch_valacc = torch.FloatTensor([0])

            for child_idx in range(self.num_of_children):
                # images, labels.cuda()

                step += 1

                model()  # forward pass without input
                sampled_architecture = model.sampled_architecture
                sampled_entropies = model.sampled_entropies.detach()
                sampled_logprobs = model.sampled_logprobs

                # get the acc of a single child
                # (sampled_architecture)
                conf = self.make_enas_config(sampled_architecture)
                self.logger.info(conf)
                if self.isShared:
                    child = self.child
                else:
                    child = EnasChild(conf, self.num_layers, self.learning_rate_child, momentum,
                                      num_classes=self.num_classes, out_filters=self.out_filters,
                                      input_shape=self.input_dim).to(device)
                self.logger.info("train_controller, epoch/child : ", epoch_idx, child_idx, " child : ", child)
                self.train_child(child, conf, device, train_loader, 1, epoch_idx, child_idx)
                validation_accuracy = self.test_child(child, conf, device, valid_loader)

                # with torch.no_grad():
                #    prediction = builder(sampled_architecture)(images)
                # validation_accuracy = torch.mean((torch.max(prediction, 1)[1] == labels).type(torch.float)) #TODO: this shoudl be changed according to the data
                # or F.nll_loss(prediction, labels, reduction = "sum").item()  #sum up batch loss

                reward = torch.tensor(validation_accuracy).detach()
                # reward += sampled_entropies * entropy_weight

                # calculate advantage with baseline (moving avg)

                baseline = prev_runs.mean()  # substract baseline to reduce variance in rewards
                reward = reward - baseline
                self.logger.info(prev_runs, baseline, reward)

                loss += -1 * sampled_logprobs * reward
                epoch_valacc += validation_accuracy

                # logging to tensorboard
                self.writer.add_scalar("loss", loss.item(), global_step=step)
                self.writer.add_scalar("reward", reward, global_step=step)
                self.writer.add_scalar("valid_acc", validation_accuracy, global_step=step)

            self.logger.info("step:", step)

            self.writer.add_scalar("entropy_weight", entropy_weight, global_step=epoch_idx)
            self.writer.add_histogram("sampled_arcs", model.sampled_architecture, global_step=epoch_idx)
            self.writer.add_scalar("sampled_logprobs", model.sampled_logprobs, global_step=epoch_idx)
            self.writer.add_scalar("sampled_entropies", model.sampled_entropies, global_step=epoch_idx)

            loss /= self.num_of_children
            epoch_valacc /= self.num_of_children

            # trainig:
            loss.backward(retain_graph=True)  # retrain_graph: keep the gradients, idk if we need this but tdvries does
            # to normalize gradients : grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradBound) #normalize gradient
            optimizer.step()
            model.zero_grad()

            self.writer.add_scalar("epoch_loss", loss.item(), global_step=epoch_idx)
            self.writer.add_scalar("epoch_loss", epoch_valacc, global_step=epoch_idx)

            prev_runs = push_to_tensor_alternative(prev_runs, epoch_valacc)

        return prev_runs

    def train_child(self, child, config, device, train_loader, epochs, c_epoch_idx, child_idx):

        child.train()
        for epoch_idx in range(epochs):
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                child.optimizer.zero_grad()
                prediction = child(images, config)
                loss = F.nll_loss(prediction, labels)
                loss.backward()
                child.optimizer.step()

                if batch_idx % self.log_interval == 0:
                    self.logger.info('Train Epoch: {}-{}-{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        c_epoch_idx, child_idx, epoch_idx
                        , batch_idx * len(images), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    def test_child(self, child, config, device, valid_loader):

        child.eval()
        validation_loss = 0
        correct = 0
        n = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                prediction_probs = child(images, config)
                validation_loss += F.nll_loss(prediction_probs, labels, reduction="sum").item()  # batch loss
                prediction = prediction_probs.argmax(dim=1, keepdim=True)  # index of max logprob
                correct += prediction.eq(labels.view_as(prediction)).sum().item()
                # n += len(prediction)

        validation_loss /= len(valid_loader.dataset)

        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset)))

        return 100. * correct / len(valid_loader.dataset)
