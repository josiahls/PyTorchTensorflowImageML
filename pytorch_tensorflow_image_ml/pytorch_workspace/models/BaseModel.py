from abc import ABC, abstractmethod

from torch import nn

from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.dataset_mnist_pytorch import BasePyTorchDataset


class BaseModel(ABC):
    def __init__(self, config: Config, dataset: BasePyTorchDataset):
        self.config = config
        self.dataset = dataset
        self.model = None  # type: nn.Module

    def run_n_epochs(self, epochs: int):
        pass

    @abstractmethod
    def step(self):
        pass
