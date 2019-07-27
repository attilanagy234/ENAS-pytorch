from net_manager import NetManager
from child import Child
from torchvision import datasets, transforms
import torch

if __name__ == "__main__":
    '''
    data_loader = DataLoader(path)
    data = dataLoader.load_data()
    net_manager = NetManager()
    net_manager.start(data)
    '''

    batch_size = 64
    learning_rate = 0.01
    momentum = 0.5
    l2_decay = 0

    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")


    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True)

    param_per_layer = 4
    num_of_layers = 2
    input_dim = (28, 28)
    netmanager = NetManager(5, 5, 1, 5, 5, param_per_layer, num_of_layers, 5)

    # kernel_size stride pooling_size  output_dim'
    conf = netmanager.make_config("5 1 5 20     5 1 5 50")
    child = Child(conf, learning_rate, momentum, 10, (28, 28))
    print(child)
    for epoch_idx in range(4):
        netmanager.train_child(child, device, train_loader, 1)
        netmanager.test_child(child, device, test_loader)

