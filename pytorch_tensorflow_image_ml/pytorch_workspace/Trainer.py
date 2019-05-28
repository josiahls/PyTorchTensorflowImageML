from typing import List

from torch.utils.data import DataLoader

from pytorch_tensorflow_image_ml.pytorch_workspace.models.BaseModel import BaseModel
from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.datasets_pytorch import BasePyTorchDataset


class Trainer(object):

    def __init__(self, config: Config, models, datasets: List[BasePyTorchDataset]):
        """
        Handles model iteration and datasets iteration.
        Allows for a quick convenient way to test many models.

        Args:
            config:
            models:
            datasets:
        """
        self.config = config
        self.models = models
        self.datasets = datasets

    def run_model_on_datasets(self, model: type(BaseModel), datasets: List[BasePyTorchDataset]):
        """
        Iterates through the list of datasets given a model,
        and trains the model on each dataset.

        Will do k-fold cross validation.

        Args:
            model:
            datasets:

        Returns:

        """
        for dataset in datasets:

            for k in range(self.config.k_folds):
                model_instance = model(self.config, dataset)
                train_val_results = model_instance.run_n_epochs(self.config.epochs, self.config.validate_train_split)
                dataset.toggle_test_train(True)
                test_results = model_instance.step(DataLoader(dataset=dataset,
                                                              batch_size=self.config.batch_size, shuffle=True), True)
                dataset.toggle_test_train(False)

    def run_models_on_datasets(self):
        """
        Primarily iterates through the `self.models` field, and tests each model
        on the list of datasets.

        Returns:

        """
        for model in self.models:
            self.run_model_on_datasets(model, self.datasets)
