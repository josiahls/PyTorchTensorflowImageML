import torch
from torch import nn, optim
from pytorch_tensorflow_image_ml.pytorch_workspace.models.BaseModel import BaseModel
from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.datasets_pytorch import BasePyTorchDataset
import numpy as np


class LinearModel(BaseModel):
    NAME = 'LinearModel'

    def __init__(self, config: Config, dataset: BasePyTorchDataset):
        super().__init__(config, dataset)

        # Input shape will need to be flattened
        self.input_shape = self.input_shape if len(self.input_shape) == 1 else np.prod(self.input_shape)
        self.output_shape = self.output_shape if len(self.output_shape) == 1 else np.prod(self.output_shape)

        # Init the module
        # noinspection PyUnresolvedReferences
        self.model = LinearNNModule(torch.empty(self.input_shape),
                                    torch.empty(self.output_shape)).to(device=self.device)

        # To keep the base model clean, when calling the loss function, we modify the y to match the format the function
        # is expecting.
        self.criterion = None
        self.optimizer = optim.SGD(self.model.parameters(), self.config.learning_rate, self.config.momentum)

        # Go through the model layers and add forward hooks to them
        # noinspection PyProtectedMember
        [getattr(self.model, layer).register_forward_hook(self.get_activation(layer)) for layer in self.model._modules]


# noinspection PyUnresolvedReferences
class LinearNNModule(nn.Module):
    def __init__(self, input_tensor: torch.tensor, output_tensor: torch.tensor):
        super().__init__()

        # Set up the output layers.
        self.f1 = nn.Linear(input_tensor.unsqueeze(0).shape[1], input_tensor.unsqueeze(0).shape[1])
        self.out = nn.Linear(input_tensor.unsqueeze(0).shape[1], output_tensor.unsqueeze(0).shape[1])

    def forward(self, *input_tensor):
        x = nn.LeakyReLU()(self.f1(*input_tensor))
        return self.out(x)
