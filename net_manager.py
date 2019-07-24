import torch

from controller import Controller
from builder import Builder
import torch.nn.functional as F


class NetManager(object):

    def __init__(self, log_interval, num_of_children):
        self.log_interval = log_interval
        self.children = list()
        self.num_of_children = num_of_children
        self.builder = Builder()
        self.controller = Controller()




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
