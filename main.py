import torch
import numpy as np
from tensorboardX import SummaryWriter
import datetime
from trainer import *
from utils import *
from pathlib import Path
import time
# Install latest Tensorflow build
# from tensorflow import summary


if __name__ == "__main__":
    # Seed for reproductivity
    torch.manual_seed(33)
    np.random.seed(360)

    current_time = datetime.datetime.now()
    time = str(time.time())
    logname =  ""



    CIFAR = False      #flag for mnist of cifar

    # Hyperparameters
    log_interval = 211
    learning_rate_child = 0.001
    learning_rate_controller = 0.001
    momentum = 0.8
    l2_decay = 0
    entropy_weight = 0.0001  # to encourage exploration

    epoch_controller = 100
    epoch_child = 4
    child_retrain_epoch = 10  #after each controller epoch, retraining the best performing child configuration from sratch for this many epoch
    controller_size = 5
    controller_layers = 2

    num_of_branches = 6
    num_of_layers = 6
    num_of_children = 5

    batch_size = 64
    batch_size_test = 1000
    reduced_labels = []  # other labels needs to be transformed if u skip a label
    input_dim = (28, 28)
    num_classes = 10
    out_filters = 24
    input_channels = 1

    # Data
    #train_loader, test_loader = get_data_loaders(batch_size, 1000, reduced_labels)

    if CIFAR:
        train_loader, test_loader = load_CIFAR(batch_size, 1000, [])
        input_channels=3
        out_filters = 64

        logname += "CIFAR"
    else:
        train_loader, test_loader = load_MNIST(batch_size, 1000, reduced_labels)
        input_channels=1
        logname += "MNIST"


    # command: tensorboard --logdir=runs
    writer = SummaryWriter(comment=logname)


    # Device
    use_cuda = True

    device = torch.device("cuda" if use_cuda else "cpu")

    trainer = Trainer(writer,
                      log_interval=log_interval,
                      num_of_children=num_of_children,
                      input_channels=input_channels,
                      input_shape=input_dim[0],
                      num_classes=num_classes,
                      learning_rate_child=learning_rate_child,
                      momentum_child=momentum,
                      num_branches=num_of_branches,
                      num_of_layers=num_of_layers,
                      out_filters=out_filters,
                      controller_size=controller_size,
                      controller_layers=controller_layers,
                      isShared=True
                      )

    controller_optimizer = torch.optim.Adam(params=trainer.controller.parameters(),
                                            lr=learning_rate_controller,
                                            betas=(0.0, 0.999),
                                            eps=1e-3)

    val_acc = trainer.train_controller(trainer.controller,
                                       controller_optimizer,
                                       device,
                                       train_loader,
                                       test_loader,
                                       epoch_controller,
                                       momentum,
                                       entropy_weight,
                                       child_retrain_epoch)

    writer.add_text( "hparams", str({"batch_size": batch_size,
                        "learning_rate_child": learning_rate_child,
                        "learning_rate_controller": learning_rate_controller,
                        "momentum": momentum,
                        "l2_decay": l2_decay,
                        "param_per_layer": num_of_branches,
                        "num_of_layers": num_of_layers,
                        "epoch_controller": epoch_controller,
                        "controller_size": controller_size,
                        "controller_layers": controller_layers,
                        "num_of_children": num_of_children,
                        "out_filters": out_filters,
                        "epoch_child": epoch_child,
                        "entropy_weight": entropy_weight,
                        "log_interval": log_interval,
                        "input_channels": input_channels,
                        "input_dim": input_dim
                                     }))


    writer.close()
