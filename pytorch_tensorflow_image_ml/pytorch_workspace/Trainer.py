from typing import List
import numpy as np
from torch.utils.data import DataLoader

from pytorch_tensorflow_image_ml.pytorch_workspace.models.BaseModel import BaseModel
from pytorch_tensorflow_image_ml.utils.PytorchSummaryWriter import PyTorchSummaryWriter
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
            avg_test_writer = PyTorchSummaryWriter(f'_test_{model.NAME}_{dataset.name}')
            avg_train_writer = PyTorchSummaryWriter(f'_train_{model.NAME}_{dataset.name}_averaged')
            avg_validation_writer = PyTorchSummaryWriter(f'_validation_{model.NAME}_{dataset.name}_averaged')
            k_train_val_results = []
            k_test_results = []

            for k in range(self.config.k_folds):
                train_writer = PyTorchSummaryWriter(f'_train_{model.NAME}_{dataset.name}_k_{k}')
                validation_writer = PyTorchSummaryWriter(f'_validation_{model.NAME}_{dataset.name}_k_{k}')

                model_instance = model(self.config, dataset)
                train_val_results = model_instance.run_n_epochs(self.config.epochs, self.config.validate_train_split,
                                                                train_writer, validation_writer)
                dataset.toggle_test_train(True)
                test_results = model_instance.step(DataLoader(dataset=dataset,
                                                              batch_size=self.config.batch_size, shuffle=True), True)
                dataset.toggle_test_train(False)

                k_train_val_results.append(train_val_results)
                k_test_results.append(test_results)

                train_writer.close()
                validation_writer.close()

            # Log the averages
            for i in range(len(k_train_val_results[0])):
                # Log the averages over k folds for validation
                [avg_train_writer.add_scalar(key, np.average([k_train_val_results[j][i]['validation'][key]
                                                         for j in range(len(k_train_val_results))]), i)
                 for key in k_train_val_results[0][i]['validation']]
                # Log the averages over k folds for train
                [avg_train_writer.add_scalar(key, np.average([k_train_val_results[j][i]['train'][key]
                                                         for j in range(len(k_train_val_results))]), i)
                 for key in k_train_val_results[0][i]['train']]
                # Log the averages over k folds for test (should be a giant straight line)
                [avg_test_writer.add_scalar(key, np.average([k_test_results[j][key] for j in range(len(k_test_results))]), i)
                 for key in k_test_results[0]]

            avg_train_writer.close()
            avg_validation_writer.close()
            avg_test_writer.close()

    def run_models_on_datasets(self):
        """
        Primarily iterates through the `self.models` field, and tests each model
        on the list of datasets.

        Returns:

        """
        for model in self.models:
            self.run_model_on_datasets(model, self.datasets)
