from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from . import Data


class Mnist(Data):
    def __init__(self,
                 n_clients: int,
                 batch_size: int,
                 path: str,
                 alpha: float = 100,
                 plot_path=None,
                 flag=False):
        super().__init__(plot_path)
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        self.train_set = MNIST(root=path, train=True, download=True,transform=transform)
        self.validxate_set = MNIST(root=path, train=False, transform=transform)
        self.trainLoader, self.client_nums, self.total = \
            self.train_loader(alpha, n_clients, batch_size, flag)
        self.validxationLoader = self.validxate_loader(batch_size * n_clients)

    def __str__(self):
        return "MNIST"
