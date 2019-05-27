import torch

from torch import nn

from pytorch_tensorflow_image_ml.pytorch_workspace.models.BaseModel import BaseModel
from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.dataset_mnist_pytorch import BasePyTorchDataset


class LinearModel(BaseModel):

    def __init__(self, config: Config, dataset: BasePyTorchDataset):
        super().__init__(config, dataset)
        # Init the module
        self.model = LinearNNModule(dataset[0]['x'], dataset[0]['y'])

    def step(self):
        pass


# noinspection PyUnresolvedReferences
class LinearNNModule(nn.Module):
    def __init__(self, input_tensor: torch.tensor, output_tensor: torch.tensor):
        super().__init__()

        # Set up the output layers.
        self.f1 = nn.Linear(input_tensor.unsqueeze(0).shape[1], output_tensor.unsqueeze(0).shape[1])
        self.out = nn.Linear(output_tensor.unsqueeze(0).shape[1], output_tensor.unsqueeze(0).shape[1])

    def forward(self, *input_tensor):
        x = nn.Tanh()(self.f1(*input_tensor))
        return self.out(x)
