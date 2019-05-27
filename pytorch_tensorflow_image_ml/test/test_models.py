import torch

from pytorch_tensorflow_image_ml.pytorch_workspace.models.LinearModel import LinearModel
from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.dataset_mnist_pytorch import DatasetMNIST


def test_pytorch_linear():
    config = Config()
    dataset = DatasetMNIST(n_rows=50)
    model = LinearModel(config, dataset)

    # Test dataset init
    assert torch.eq(model.dataset[0]['y'], 0)
    assert len(model.dataset[0]['x']) == 784

    model.model.forward(model.dataset[0]['x'].unsqueeze(0))


