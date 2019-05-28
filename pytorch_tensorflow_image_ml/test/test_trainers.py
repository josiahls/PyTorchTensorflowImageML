from torchvision.transforms import Compose

from pytorch_tensorflow_image_ml.pytorch_workspace.Trainer import Trainer
from pytorch_tensorflow_image_ml.pytorch_workspace.models.LinearModel import LinearModel
from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.datasets_pytorch import DatasetMNIST, DatasetFashionMNIST
from pytorch_tensorflow_image_ml.utils.transforms import MNISTToXY, ToTensor, TreatCategorical


def test_pytorch_trainer_init():
    config = Config()
    config.epochs = 20

    datasets = [
        DatasetMNIST(n_rows=100, transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()])),
        DatasetFashionMNIST(n_rows=100, transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]))
    ]

    models = [LinearModel]

    trainer = Trainer(config, models=models, datasets=datasets)
    trainer.run_models_on_datasets()
    assert True
