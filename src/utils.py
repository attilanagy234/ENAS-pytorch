import logging
import sys

import torch
from torchvision import datasets, transforms


# FIFO for baseline
def queue(tensor, x):
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


def load_MNIST(train_bs, test_bs, labels):
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

def load_CIFAR(train_bs, test_bs, labels):

    # Preprocess:
    #   -50k train / 10k test
    #   -substract channel mean/dividing by standard deviation
    #   -centrally padding the training images to 40x40 and randomly cropping them back to 30x30
    #   -randomly flipping them horizontally

    train_set = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform= transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ]))

    test_set = datasets.CIFAR10(root='./data', train=False, download=True,
                                        transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ]))

    if len(labels) != 0:
        reduceLabels(train_set, labels)

    if len(labels) != 0:
        reduceLabels(test_set, labels)


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_bs, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bs, shuffle=False, num_workers=2)

    return train_loader, test_loader

#RICAP: https://arxiv.org/abs/1811.09030


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])