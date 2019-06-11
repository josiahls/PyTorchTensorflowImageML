from functools import partial

from torchvision.transforms import Compose

from pytorch_tensorflow_image_ml.pytorch_workspace.Trainer import Trainer
from pytorch_tensorflow_image_ml.pytorch_workspace.models.LinearModel import LinearModel
from pytorch_tensorflow_image_ml.utils.callbacks import TensorboardCallback
from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.datasets_pytorch import DatasetMNIST, DatasetFashionMNIST
from pytorch_tensorflow_image_ml.utils.transforms import MNISTToXY, ToTensor, TreatCategorical


def test_trainer_pytorch_init():
    config = Config()
    config.epochs = 20
    config.k_folds = 3

    datasets = [
        DatasetMNIST(n_rows=100, transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()])),
        DatasetFashionMNIST(n_rows=100, transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]))
    ]

    models = [LinearModel]

    trainer = Trainer(config, models=models, datasets=datasets)
    trainer.run_models_on_datasets()
    assert True


def test_trainer_pytorch_run_model_on_dataset():
    config = Config()
    config.epochs = 10
    config.k_folds = 1
    trainer = Trainer(config, callbacks=[partial(TensorboardCallback, writer_prefix='callback_obj_testing')])

    result_1 = []
    result_2 = []
    dataset = DatasetMNIST(n_rows=100, transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]))
    trainer.run_model_on_dataset(LinearModel, dataset, result_1, result_2)
