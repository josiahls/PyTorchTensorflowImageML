from torch.utils.data import Dataset
from typing import List, Tuple

from pytorch_tensorflow_image_ml.pytorch_workspace.models.BaseModel import BaseModel
from pytorch_tensorflow_image_ml.utils.config import Config


class Trainer(object):

    def __init__(self, config: Config, models, datasets: List[Tuple[Dataset, Dataset]]):
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

    def run_model_on_datasets(self, model: type(BaseModel), datasets):
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
                model_instance = model(self.config, dataset[0])
                results = model_instance.run_n_epochs(self.config.epochs, self.config.validate_train_split)
                test_results = model_instance.validate(dataset[1])

    def run_models_on_datasets(self):
        """
        Primarily iterates through the `self.models` field, and tests each model
        on the list of datasets.

        Returns:

        """
        for model in self.models:
            self.run_model_on_datasets(model, self.datasets)
