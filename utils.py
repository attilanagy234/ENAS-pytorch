import torch
from torchvision import datasets, transforms


def reduceLabels(data, labels):
    """inplace!"""

    idx = data.targets == labels[0]
    for label in labels[1:]:
        label_idx = data.targets == label
        idx += label_idx
        print(label, label_idx, len(label_idx))
    data.targets = data.targets[idx]
    data.data = data.data[idx]


def get_dataLoaders(train_bs, test_bs, labels):

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

