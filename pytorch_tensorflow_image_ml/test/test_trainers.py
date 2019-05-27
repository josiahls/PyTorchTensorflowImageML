from pytorch_tensorflow_image_ml.pytorch_workspace.Trainer import Trainer
from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.dataset_mnist_pytorch import DatasetMNIST


def test_pytorch_trainer_init():
    config = Config()
    dataset = DatasetMNIST(n_rows=100)

    trainer = Trainer(config, None, dataset)
