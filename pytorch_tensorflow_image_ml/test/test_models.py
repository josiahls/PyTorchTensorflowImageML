import torch
from torchvision.transforms import Compose

from pytorch_tensorflow_image_ml.pytorch_workspace.models.LinearModel import LinearModel
from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.datasets_pytorch import DatasetMNIST
from pytorch_tensorflow_image_ml.utils.transforms import MNISTToXY, ToTensor, TreatCategorical


def test_pytorch_linear():
    config = Config()
    config.validate_train_split = False
    dataset = DatasetMNIST(n_rows=50, transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]))
    model = LinearModel(config, dataset)

    # Test dataset init
    assert torch.eq(model.dataset[0]['y'], 0)
    assert len(model.dataset[0]['x']) == 784

    model.forward(model.dataset[0]['x'].unsqueeze(0))
    model.run_n_epochs(150)
    assert torch.eq(model.forward(model.dataset[0]['x'].unsqueeze(0)).argmax(), 0)

