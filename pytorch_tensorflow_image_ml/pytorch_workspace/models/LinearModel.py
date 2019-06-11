import torch

from torch import nn, optim

from pytorch_tensorflow_image_ml.pytorch_workspace.models.BaseModel import BaseModel
from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.datasets_pytorch import BasePyTorchDataset


class LinearModel(BaseModel):
    NAME = 'LinearModel'

    def __init__(self, config: Config, dataset: BasePyTorchDataset):
        super().__init__(config, dataset)
        # Init the module
        # noinspection PyUnresolvedReferences
        self.model = LinearNNModule(dataset[0]['x'], torch.empty(dataset.n_classes)).to(device=self.device)

        # To keep the base model clean, when calling the loss function, we modify the y to match the format the function
        # is expecting.
        self.criterion = lambda y_pred, y: nn.CrossEntropyLoss()(y_pred, y.squeeze())
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
