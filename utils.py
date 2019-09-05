import logging
import sys

import torch
from torchvision import datasets, transforms


# FIFO for baseline
def push_to_tensor_alternative(tensor, x):
    return torch.cat((tensor[1:], torch.Tensor([x])))


def reduceLabels(data, labels):
    """inplace!"""

    idx = data.targets == labels[0]
    for label in labels[1:]:
        label_idx = data.targets == label
        idx += label_idx
        # print(label, label_idx, len(label_idx))
    data.targets = data.targets[idx]
    data.data = data.data[idx]


def get_data_loaders(train_bs, test_bs, labels):
    data = datasets.MNIST('../data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))

    if len(labels) != 0:
        reduceLabels(data, labels)

    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=train_bs,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        data,
        batch_size=test_bs,
        shuffle=True)

    return train_loader, test_loader


def get_logger(name=__file__, level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def get_data_loaders_CIFAR(train_bs, test_bs, labels):
    data = datasets.CIFAR10('../data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]))

    if len(labels) != 0:
        reduceLabels(data, labels)

    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=train_bs,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        data,
        batch_size=test_bs,
        shuffle=True)

    return train_loader, test_loader



