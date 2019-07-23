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

    def train_children(self, device, data_loader, optimizer, epoch):
        for child in self.children:
            child.train()
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = child(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(data_loader.dataset),
                               100. * batch_idx / len(data_loader), loss.item()))

    def get_losses(self, device, data_loader):
        for child in self.children:
            child.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in data_loader:
                    data, target = data.to(device), target.to(device)
                    output = child(data)
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(data_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(data_loader.dataset),
                100. * correct / len(data_loader.dataset)))
