from torch.utils.data import Dataset
from typing import List

from pytorch_tensorflow_image_ml.pytorch_workspace.models.BaseModel import BaseModel
from pytorch_tensorflow_image_ml.utils.config import Config


class Trainer(object):

    def __init__(self, config: Config, models, datasets: List[Dataset]):
        self.config = config
        self.models = models
        self.datasets = datasets

    def run_model_on_datasets(self, model: type(BaseModel), datasets):
        """
        Iterates through the list of datasets given a model,
        and trains the model on each dataset.

        Args:
            model:
            datasets:

        Returns:

        """
        for dataset in datasets:
            model_instance = model(self.config, dataset)
            model_instance.run_n_epochs(self.config.epochs)

    def run_models_on_datasets(self):
        """
        Primarily iterates through the `self.models` field, and tests each model
        on the list of datasets.

        Returns:

        """
        for model in self.models:
            self.run_model_on_datasets(model, self.datasets)
