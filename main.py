from net_manager import NetManager
from child import Child
import torch
from torchvision import datasets, transforms
import torch
import numpy as np
from tensorboardX import SummaryWriter
import datetime

# Install latest Tensorflow build
from tensorflow import summary


def get_dataLoaders(train_bs, test_bs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_bs, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_bs, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    # Seed for reproductivity
    torch.manual_seed(69)
    np.random.seed(360)

    currenttime = datetime.datetime.now()
    writer = SummaryWriter("runs/" + str(currenttime))


    # Hyperparameters
    batch_size = 100
    learning_rate_child = 0.01
    learning_rate_controller = 0.01
    momentum = 0.5
    l2_decay = 0
    param_per_layer = 4
    num_of_layers = 2
    input_dim = (28, 28)
    num_of_children = 3
    epoch_controller = 3
    epoch_child = 1
    entropy_weight = 0.1  # to encourage exploration
    loginterval = 5
    input_channels = 1
    output_dim = 10
    controller_size = 5
    controller_layers = 2

    # Data
    train_loader, test_loader = get_dataLoaders(batch_size, 1000)

    # Device
    use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")

    net_manager = NetManager(writer,
                             loginterval,
                             num_of_children,
                             input_channels,
                             output_dim,
                             learning_rate_child,
                             param_per_layer,
                             num_of_layers,
                             10,  # =out_filters, not used
                             controller_size,
                             controller_layers
                             )

    controller_optimizer = torch.optim.Adam(params=net_manager.controller.parameters(),
                                            lr=learning_rate_controller,
                                            betas=(0.0, 0.999),
                                            eps=1e-3)

    net_manager.train_controller(net_manager.controller,
                                 controller_optimizer,
                                 device,
                                 train_loader,
                                 test_loader,
                                 epoch_controller,
                                 momentum,
                                 entropy_weight)

    writer.close()