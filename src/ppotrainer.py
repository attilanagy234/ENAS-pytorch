from typing import Dict, Union
import torch
import torch.nn.functional as F
from collections import namedtuple
# from controller import Controller
from PPOController import *
# from child import Child
from EnasChild import *
from utils import queue, get_logger
from kindergarden import *
import numpy as np

layer = namedtuple('layer', 'kernel_size stride pooling_size input_dim output_dim')


class PPOTrainer(object):

    def __init__(self,
                 writer,
                 log_interval,
                 num_of_children,
                 input_channels,
                 input_shape,
                 num_classes,
                 learning_rate_child,
                 momentum_child,
                 num_branches,
                 num_of_layers,
                 out_filters,
                 controller_size,
                 controller_layers,
                 t0,
                 eta_min,
                 t_mult,
                 epoch_child,
                 isShared,
                 eps_clip=0.2,
                 K_epochs=10,
                 path = ""):

        self.writer = writer
        self.log_interval = log_interval
        self.path=path

        self.num_of_children = num_of_children
        self.K_epochs = K_epochs

        self.input_shape = input_shape
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.out_filters = out_filters

        self.isShared = isShared

        self.epoch_child = epoch_child

        self.num_branches = num_branches
        self.num_layers = num_of_layers

        self.learning_rate_child = learning_rate_child
        self.momentum = momentum_child

        self.t0 = t0
        self.eta_min = eta_min
        self.t_mult = t_mult

        self.eps_clip = eps_clip

        self.controller = PPOController(self.writer,
                                        self.num_layers,
                                        self.num_branches,
                                        controller_size,
                                        controller_layers)  # self,num_layers=2,num_branches=4,lstm_size=5,lstm_num_layers=2,tanh_constant=1.5,temperature=None

        self.controller_old = PPOController(self.writer,
                                            self.num_layers,
                                            self.num_branches,
                                            controller_size,
                                            controller_layers)  # self,num_layers=2,num_branches=4,lstm_size=5,lstm_num_layers=2,tanh_constant=1.5,temperature=None

        self.controller.load_state_dict(self.controller_old.state_dict())

        self.children = list()

        self.bestchilds = Kindergarden(best_of=3)


        self.globaliter = 0
        self.logger = get_logger()

        if self.isShared:
            self.child = SharedEnasChild(self.num_layers, self.learning_rate_child, self.momentum,
                                         num_classes=self.num_classes, out_filters=self.out_filters,
                                         input_shape=self.input_shape, input_channels=self.input_channels,
                                         t0=t0, eta_min=eta_min
                                         )

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
        return raw_config

    # unused
    def sample_child(self):
        return self.controller.sample()

    # unused
    def sample_children(self):
        for i in range(self.num_of_children):
            self.children.append(self.controller.sample())

    def train_controller(self, old_model, new_model, optimizer, device, train_loader, valid_loader, epoch, momentum,
                         entropy_weight, child_retrain_epoch, child_retrain_interval):

        new_model.train()

        step = 0

        prev_runs = torch.zeros([5])  # to store the val_acc of prev epochs

        for epoch_idx in range(epoch):
            loss = torch.FloatTensor([0])

            epoch_valacc = torch.zeros(self.num_of_children)
            epoch_childs = []

            for child_idx in range(self.num_of_children):
                # images, labels.cuda()

                step += 1

                old_model()  # forward pass without input
                sampled_architecture = old_model.sampled_architecture
                sampled_entropies = old_model.sampled_entropies.detach()
                sampled_logprobs = old_model.sampled_logprobs

                # get the acc of a single child

                # make child
                conf = self.make_enas_config(sampled_architecture)
                epoch_childs.append(conf)

                print("CONF:", conf)

                if self.isShared:
                    child = self.child.to(device)
                else:
                    child = SharedEnasChild(conf, self.num_layers, self.learning_rate_child, momentum,
                                            num_classes=self.num_classes, out_filters=self.out_filters,
                                            input_shape=self.input_shape, input_channels=self.input_channels).to(device)

                #               self.logger.info("train_controller, epoch/child : ", epoch_idx, child_idx, " child : ", conf) # logging error

                # Train child
                self.train_child(child, conf, device, train_loader, self.epoch_child, epoch_idx, child_idx)

                # Test child
                validation_accuracy, validation_loss = self.test_child(child, conf, device, valid_loader)
                epoch_valacc[child_idx] = validation_accuracy
                self.bestchilds.add(conf, validation_accuracy)

                reward = torch.tensor(validation_accuracy).detach()
                reward += sampled_entropies * entropy_weight

                # calculate advantage with baseline (moving avg)

                baseline = prev_runs.mean()  # substract baseline to reduce variance in rewards
                reward = reward - baseline

                old_model.memory.add_rewards(reward)

                # logging to tensorboard
                self.writer.add_scalar("reward", reward, global_step=step)
                self.writer.add_scalar("valid_acc", validation_accuracy, global_step=step)
                self.writer.add_scalar("valid_loss", validation_loss, global_step=step)
                self.writer.add_scalar("sampled_entropies", sampled_entropies, global_step=step)
                self.writer.add_scalar("sampled_logprobs", sampled_logprobs, global_step=step)

            best_child_idx = torch.argmax(epoch_valacc)
            best_child_conf = epoch_childs[best_child_idx]

            message = " best valacc" + str(epoch_valacc[best_child_idx].item())\
                      + ' - config: ' + str(best_child_conf)

            self.writer.add_text("best child", message, global_step=epoch_idx)

            if epoch_idx % child_retrain_interval == 0:

                retrained_valacc, retrained_loss = self.retrain(best_child_conf, device, train_loader, valid_loader, child_retrain_epoch, epoch_idx)

                print("current best childs: ", self.bestchilds.bestchilds)
                self.writer.add_scalar("retrainerd child valacc", retrained_valacc, epoch_idx)

                self.save(epoch_idx)

            old_branch_logprobs, old_skip_logprobs = old_model.memory.get_logprobs()
            old_rewards = old_model.memory.rewards

            old_skip_logprobs = torch.tensor(old_skip_logprobs, dtype=torch.float).detach()  # NO GRADIENT Info
            old_branch_logprobs = torch.tensor(old_branch_logprobs, dtype=torch.float).detach()  # NO GRADIENT Info

            # #Optimize policy for K epochs:

            for _ in range(self.K_epochs):

                #print("inside ppo update loop")

                new_branch_logprobs = torch.zeros_like(old_branch_logprobs)
                new_skip_logprobs = torch.zeros_like(old_skip_logprobs)
                branch_entropies = torch.zeros_like(old_skip_logprobs)
                skip_entropies = torch.zeros_like(old_skip_logprobs)

                #print("branchhsape", new_branch_logprobs.shape)
                #print("skiphape", new_skip_logprobs.shape)
                #print("branch entropyshape", branch_entropies.shape)
                #print("skip entropyshape", skip_entropies.shape)

                for tx_idx in range(len(old_model.memory.rewards)):
                    #print(new_model.evaluate(old_model.memory.transitions[tx_idx]))
                    new_branch_logprobs[tx_idx], new_skip_logprobs[tx_idx], branch_entropies[tx_idx], skip_entropies[
                        tx_idx] = new_model.evaluate(old_model.memory.transitions[tx_idx])

                # Finding the ratio (pi_theta / pi_theta__old):
                branch_ratios = torch.exp(new_branch_logprobs - old_branch_logprobs)
                skip_ratios = torch.exp(new_skip_logprobs - old_skip_logprobs)
                ratios = torch.cat((branch_ratios, skip_ratios))

                advantages = torch.Tensor(old_rewards).mean()
                ratios = ratios.mean()
                #print( ratios, advantages)

                surr1 = ratios * advantages
                #print(surr1)

                surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
                loss = -torch.min(surr1, surr2)  # + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
                #print(loss)

                # take gradient step
                optimizer.zero_grad()
                loss.mean().backward(retain_graph=True)
                optimizer.step()

            # Copy new weights into old policy:
            old_model.memory.clean()
            old_model.load_state_dict(new_model.state_dict())

            # self.writer.add_histogram("sampled_branches", model.sampled_architecture, global_step=epoch_idx)
            # self.writer.add_histogram("sampled_connections", model.sampled_architecture[1], global_step=epoch_idx)

            self.writer.add_scalar("epoch_loss", loss.mean().item(), global_step=epoch_idx)
            self.writer.add_scalar("epoch mean validation acc.", epoch_valacc.mean(), global_step=epoch_idx)


            # self.writer.add_graph(child) #ERROR:  TracedModules don't support parameter sharing between modules

            prev_runs = queue(prev_runs, epoch_valacc.mean())

        return prev_runs

    def train_child(self, child, config, device, train_loader, epochs, c_epoch_idx, child_idx):

        child.train()

        for epoch_idx in range(epochs):
            for batch_idx, (images, labels) in enumerate(train_loader):

                images, labels = images.to(device), labels.to(device)
                child.to(device)
                child.optimizer.zero_grad()
                prediction = child(images, config)
                loss = F.nll_loss(prediction, labels)
                loss.backward()
                child.optimizer.step()
                child.scheduler.step()

                # Warm Restart child scheduler
                if child.optimizer.param_groups[0]['lr'] == self.eta_min:
                    self.t0 *= self.t_mult
                    child.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        child.optimizer, T_max=self.t0, eta_min=self.eta_min, last_epoch=-1)

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

        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.000f}%)\n'.format(
            validation_loss, correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset)))

        return 100. * correct / len(valid_loader.dataset), validation_loss

    def traintest_fixed_architecture(self, config, device, train_loader, valid_loader, train_epoch=10):

        fixed_child = FixedEnasChild(config, num_layers=self.num_layers, lr=self.learning_rate_child,
                                     momentum=self.momentum,
                                     num_classes=self.num_classes, out_filters=self.out_filters,
                                     input_shape=self.input_shape, input_channels=self.input_channels).to(device)

        self.train_child(fixed_child, config, device, train_loader, train_epoch, 0, 0)

        return self.test_child(fixed_child, config, device, valid_loader)


    def retrain(self, config, device, train_loader, valid_loader, epochs, c_epoch_idx):



        child = FixedEnasChild(config, num_layers=self.num_layers, lr=self.learning_rate_child,
                               momentum=self.momentum,
                               num_classes=self.num_classes, out_filters=self.out_filters,
                               input_shape=self.input_shape, input_channels=self.input_channels).to(device)
        child.train()

        for epoch_idx in range(epochs):

            epoch_loss = 0

            print("Retreaining child, epoch:" , epoch_idx)

            for batch_idx, (images, labels) in enumerate(train_loader):

                images, labels = images.to(device), labels.to(device)
                child.to(device)
                child.optimizer.zero_grad()
                prediction = child(images, config)
                loss = F.nll_loss(prediction, labels)
                epoch_loss += loss

                loss.backward()
                child.optimizer.step()
                child.scheduler.step()

                # Warm Restart child scheduler
                if child.optimizer.param_groups[0]['lr'] == self.eta_min:
                    self.t0 *= self.t_mult
                    child.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        child.optimizer, T_max=self.t0, eta_min=self.eta_min, last_epoch=-1)

                if batch_idx % self.log_interval == 0:
                    self.logger.info('Train Epoch: {}-{}-{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        c_epoch_idx, 0, epoch_idx
                        , batch_idx * len(images), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

            validacc, validloss = self.test_child(child, config, device, valid_loader)


            self.writer.add_scalars(main_tag="retrainded_child-" + str(c_epoch_idx),
                                    tag_scalar_dict={
                                        "trainloss": epoch_loss/batch_idx,
                                        "validloss": validloss,
                                        "validAcc ": validacc}
                                   ,global_step=epoch_idx)

        self.writer.add_scalar("child retrain valacc", validacc, epoch_idx)
        self.writer.add_scalar("child retrain valloss", validloss, epoch_idx)

        return validacc, validloss


    def save(self, epoch):
        print("model saved")
        path2 = self.path + "/" + str(epoch)
        torch.save(self.controller.state_dict(), self.path)