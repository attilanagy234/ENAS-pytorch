import argparse
import datetime
import time

import numpy as np
from tensorboardX import SummaryWriter

from trainer import *

# Install latest Tensorflow build
# from tensorflow import summary
from utils import load_CIFAR, load_MNIST

if __name__ == "__main__":

    argparse_formatter = lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=60, width=250)

    parser = argparse.ArgumentParser(formatter_class=argparse_formatter,
                                     description='Running efficient neural architecture search')

    parser.add_argument('-c', '--cuda', default=False, action='store_true', help='[OPTIONAL] Enablue the usage of Cuda')



    args = parser.parse_args()




    # Seed for reproductivity
    torch.manual_seed(33)
    np.random.seed(360)

    current_time = datetime.datetime.now()
    time = str(time.time())
    logname = ""

    # TODO: check
    # Enas hiperparams:
    # momentum = 0.5, TODO: cosine scheduling(lmax = 0.05, lmin = 0.001, T0 = 10, Tmul2)
    # architecture search run for 310 epoch
    # child :TODO: params initialized with He, l2_decay = 10-4,
    # controller : params [-0.1, 0.1], lr = 0.00035, tanh constant=2.5, temperature=5, entropy_Weight = 0.1,
    # SKIPCONNECTIONS: adding the KL div between skip prob between any to layer, + p=0.4 which is the prior belief that a skip connection is formed, this KL div weighted by 0.8
    # CNN LAYERS structure: RELU - CONV - BN
    # skip connections: if multiple skipconenctions: depthwise concat to match the channels then BR + relu
    # GLOBAL AVG POOLING BEFORE FULLY CONNECTEDS

    CIFAR = False  # flag for mnist or cifar

    # Hyperparameters
    log_interval = 211
    learning_rate_child = 0.05
    learning_rate_controller = 0.00035
    L_max = 0.05
    L_min = 0.001
    T_0 = 10
    T_mult = 2
    momentum = 0.8
    l2_decay = 0
    tanh_const = 2.5  # TODO:propagate to controller
    temperature = 5  # TODO:propagate to controller
    entropy_weight = 0.1  # to encourage exploration

    epoch_controller = 100
    epoch_child = 4
    child_retrain_epoch = 10  # after each controller epoch, retraining the best performing child configuration from sratch for this many epoch
    controller_size = 5
    controller_layers = 2

    num_valid_batch = 1  # child validation using only num_valid_batches batch TODO: implement in code

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
    # train_loader, test_loader = get_data_loaders(batch_size, 1000, reduced_labels)

    if CIFAR:
        train_loader, test_loader = load_CIFAR(batch_size, 1000, [])
        input_channels = 3
        out_filters = 64

        logname += "CIFAR"
    else:
        train_loader, test_loader = load_MNIST(batch_size, 1000, reduced_labels)
        input_channels = 1
        logname += "MNIST"

    # command: tensorboard --logdir=runs
    writer = SummaryWriter(comment=logname)

    # Device
    use_cuda = args.cuda

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
                      isShared=True,
                      t0=T_0,
                      t_mult=T_mult,
                      eta_min=L_min,
                      epoch_child=epoch_child,
                      )

    controller_optimizer = torch.optim.Adam(params=trainer.controller.parameters(),
                                            lr=learning_rate_controller,
                                            betas=(0.0, 0.999),
                                            eps=1e-3)

    params = str({"batch_size": batch_size,
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
                  })

    writer.add_text("hparams", params)
    print(params)
    val_acc = trainer.train_controller(trainer.controller,
                                       controller_optimizer,
                                       device,
                                       train_loader,
                                       test_loader,
                                       epoch_controller,
                                       momentum,
                                       entropy_weight,
                                       child_retrain_epoch)

    writer.close()
