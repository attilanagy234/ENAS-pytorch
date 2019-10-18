import torch
import numpy as np
from tensorboardX import SummaryWriter
import datetime
from trainer import *
from ppotrainer import *
from utils import *
from kindergarden import *
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




    #TODO: check
    #Enas hiperparams:
    # momentum = 0.5,  cosine scheduling(lmax = 0.05, lmin = 0.001, T0 = 10, Tmul2)
    # architecture search run for 310 epoch
    # child : l2_decay = 10-4 or 2e-4, TODO: params initialized with He,
    # controller : params [-0.1, 0.1], lr = 0.00035, tanh constant=2.5, temperature=5, entropy_Weight = 0.1,
    # SKIPCONNECTIONS: adding the KL div between skip prob between any to layer, + p=0.4 which is the prior belief that a skip connection is formed, this KL div weighted by 0.8
    # CNN LAYERS structure: RELU - CONV - BN
    # skip connections: if multiple skipconenctions: depthwise concat to match the channels then BR + relu
    # GLOBAL AVG POOLING BEFORE FULLY CONNECTEDS


    CIFAR = False      #flag for mnist of cifar
    trainPPO = False

    # Hyperparameters
    log_interval = 10
    learning_rate_child = 0.05
    learning_rate_controller = 0.00035
    L_max = 0.05
    L_min = 0.001
    T_0 = 10
    T_mult = 2
    momentum = 0.8
    l2_decay = 0.00025
    tanh_const = 2.5  # set in controller    original: 2.5 or 1.5
    temperature = 5  # set in controller
    entropy_weight = 0.1  # to encourage exploration

    epoch_controller = 310
    epoch_child = 1     # maybe: --eval_every_epochs=1 in the original code

    controller_size = 5
    controller_layers = 2

    # after each child_retrain_interval epoch, retraining the best performing child configuration from sratch for this many epoch
    child_retrain_interval = 1 #10
    child_retrain_epoch = 1 #20

    num_valid_batch = 1 # child validation using only num_valid_batches batch TODO: implement in code

# normal:   --child_num_layers = 12 child_out_filters = 36
#final:   --child_num_layers=24  --child_out_filters=96

    num_of_branches = 6
    num_of_layers = 3 #12
    num_of_children = 1 #5
    out_filters = 10


    batch_size = 64
    batch_size_test = 1000
    reduced_labels = [0, 1]  # other labels needs to be transformed if u skip a label
    input_dim = (28, 28)
    num_classes = 10
    input_channels = 1

    # Data
    #train_loader, test_loader = get_data_loaders(batch_size, 1000, reduced_labels)

    if CIFAR:
        train_loader, test_loader = load_CIFAR(batch_size, 1000, [])
        input_channels = 3
        input_dim = (32, 32)
        out_filters = 64

        logname += "CIFAR"
    else:
        train_loader, test_loader = load_MNIST(batch_size, 1000, reduced_labels)
        input_channels=1
        logname += "MNIST"


    # command: tensorboard --logdir=runs
    writer = SummaryWriter(comment=logname)

    # Device
    use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")

    params =   str({"batch_size": batch_size,
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
                    "input_dim": input_dim,
                    "CIFAR": CIFAR,
                    "PPO": trainPPO
                    })

    writer.add_text("hparams", params)
    print(params)

    if trainPPO==False:

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
        val_acc = trainer.train_controller(trainer.controller,
                                           controller_optimizer,
                                           device,
                                           train_loader,
                                           test_loader,
                                           epoch_controller,
                                           momentum,
                                           entropy_weight,
                                           child_retrain_epoch,
                                           child_retrain_interval)

    if trainPPO==True:

        trainer = PPOTrainer(writer,
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

        #kicsitodo: controllers / sharedChild should be created independently from trainer

        val_acc = trainer.train_controller(trainer.controller_old,
                                           trainer.controller,
                                           controller_optimizer,
                                           device,
                                           train_loader,
                                           test_loader,
                                           epoch_controller,
                                           momentum,
                                           entropy_weight,
                                           child_retrain_epoch,
                                           child_retrain_interval)





    writer.close()
