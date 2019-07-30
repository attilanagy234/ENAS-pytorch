from typing import Dict, Union
import torch
import torch.nn.functional as F
from collections import namedtuple
from controller import Controller
from child import Child

layer = namedtuple('layer', 'kernel_size stride pooling_size input_dim output_dim')


class NetManager(object):

    def __init__(self, writer, log_interval, num_of_children, input_dim, output_dim, learning_rate_child,
                 param_per_layer,
                 num_of_layers, out_filters, controller_size, controller_layers):

        self.writer = writer
        self.log_interval = log_interval
        self.num_of_children = num_of_children

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.param_per_layer = param_per_layer
        self.num_of_layers = num_of_layers

        self.learning_rate_child = learning_rate_child

        self.controller = Controller(writer, num_of_layers, param_per_layer, controller_size,
                                     controller_layers)  # self,num_layers=2,num_branches=4,lstm_size=5,lstm_num_layers=2,tanh_constant=1.5,temperature=None
        self.children = list()

        self.globaliter = 0

    def make_config(self, raw_config):

        config = dict()

        prev_dim = self.input_dim

        list_config = list(map(int, raw_config))

        for layer_i in range(0, self.num_of_layers * self.param_per_layer, self.param_per_layer):
            kernel_size = 3 if list_config[layer_i + 0] else 5
            stride = 1 if list_config[layer_i + 1] else 2
            pooling_size = 2 if list_config[layer_i + 2] else 2
            input_dim = prev_dim
            out_channels = 10 if list_config[layer_i + 2] else 20
            prev_dim = out_channels

            current = layer(kernel_size, stride, pooling_size, input_dim, out_channels)
            config["layer_" + str(layer_i)] = current

        return config

    def sample_child(self):
        return self.controller.sample()

    def sample_children(self):
        for i in range(self.num_of_children):
            self.children.append(self.controller.sample())

    def train_controller(self, model, optimizer, device, train_loader, valid_loader, epoch, momentum,
                         entropy_weight):

        model.train()

        step = 0
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
                print(sampled_architecture)
                conf = self.make_config(sampled_architecture)
                print(conf)
                child = Child(conf, self.learning_rate_child, momentum, 10, (28, 28)).to(device)
                print("train_controller, epoch/child : ", epoch_idx, child_idx, " child : ", child)
                self.train_child(child, device, train_loader, 1)
                validation_accuracy = self.test_child(child, device, valid_loader)

                # with torch.no_grad():
                #    prediction = builder(sampled_architecture)(images)
                # validation_accuracy = torch.mean((torch.max(prediction, 1)[1] == labels).type(torch.float)) #TODO: this shoudl be changed according to the data
                # or F.nll_loss(prediction, labels, reduction = "sum").item()  #sum up batch loss

                reward = torch.tensor(validation_accuracy).detach()
                reward += sampled_entropies * entropy_weight

                # calculate advantage with baseline (moving avg)

                loss += -sampled_logprobs * reward
                epoch_valacc += validation_accuracy


                # logging to tensorboard
                self.writer.add_scalar("loss", loss.item())
                self.writer.add_scalar("reward", reward)
                self.writer.add_scalar("valid_acc", validation_accuracy)



            self.writer.add_scalar("entropy_weight", entropy_weight)
            self.writer.add_histogram("sampled_arc", model.sampled_architecture)
            self.writer.add_scalar("sampled_logprobs", model.sampled_logprobs)
            self.writer.add_scalar("sampled_entropies", model.sampled_entropies)

            loss /= self.num_of_children
            epoch_valacc /= self.num_of_children

            loss.backward(retain_graph=True)  # retrain_graph: keep the gradients, idk if we need this but tdvries does

            # to normalize gradients : grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradBound) #normalize gradient
            optimizer.step()
            model.zero_grad()

            self.writer.add_scalar("epoch_loss", loss.item(), global_step=epoch_idx)
            self.writer.add_scalar("epoch_loss", epoch_valacc, global_step=epoch_idx)

        return epoch_valacc

    def train_child(self, child, device, train_loader, epochs, ):

        child.train()
        for epoch_idx in range(epochs):
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                child.optimizer.zero_grad()
                prediction = child(images)
                loss = F.nll_loss(prediction, labels)
                loss.backward()
                child.optimizer.step()

                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch_idx, batch_idx * len(images), len(train_loader.dataset),
                                   100. * batch_idx / len(train_loader), loss.item()))

    def test_child(self, child, device, valid_loader):

        child.eval()
        validation_loss = 0
        correct = 0
        n = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                prediction_probs = child(images)
                validation_loss += F.nll_loss(prediction_probs, labels, reduction="sum").item()  # batch loss
                prediction = prediction_probs.argmax(dim=1, keepdim=True)  # index of max logprob
                correct += prediction.eq(labels.view_as(prediction)).sum().item()
                # n += len(prediction)

        validation_loss /= len(valid_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset)))

        return 100. * correct / len(valid_loader.dataset)
