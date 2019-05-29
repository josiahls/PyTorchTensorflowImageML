from typing import List
import numpy as np
from torch.utils.data import DataLoader

from pytorch_tensorflow_image_ml.pytorch_workspace.models.BaseModel import BaseModel
from pytorch_tensorflow_image_ml.utils.PytorchSummaryWriter import PyTorchSummaryWriter
from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.datasets_pytorch import BasePyTorchDataset


class Trainer(object):

    def __init__(self, config: Config, models=None, datasets: List[BasePyTorchDataset]=None, writer_prefix=''):
        """
        Handles model iteration and datasets iteration.
        Allows for a quick convenient way to test many models.

        Args:
            writer_prefix (str): A global prefix to add to all the tensorboard writer outputs.
            Useful for differentiating unit testing runs from actual runs.
            config:
            models:
            datasets:
        """
        self.w_p = writer_prefix
        self.config = config
        self.models = models
        self.datasets = datasets

    def run_model_on_dataset(self, model, dataset, k_train_val_results: list,
                             k_test_results: list):
        """
        Run k-fold cross val on a single model and a single dataset.

        Args:
            model:
            dataset:
            k_train_val_results:
            k_test_results:

        Returns:

        """
        for k in range(self.config.k_folds):
            train_writer = PyTorchSummaryWriter(f'{self.w_p}_train_{model.NAME}_{dataset.name}_k_{k}')
            validation_writer = PyTorchSummaryWriter(f'{self.w_p}_validation_{model.NAME}_{dataset.name}_k_{k}')

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
            avg_test_writer = PyTorchSummaryWriter(f'{self.w_p}_test_{model.NAME}_{dataset.name}_averaged')
            avg_train_writer = PyTorchSummaryWriter(f'{self.w_p}_train_{model.NAME}_{dataset.name}_averaged')
            avg_validation_writer = PyTorchSummaryWriter(f'{self.w_p}_validation_{model.NAME}_{dataset.name}_averaged')
            k_train_val_results = []
            k_test_results = []

            self.run_model_on_dataset(model, dataset, k_train_val_results, k_test_results)

            # Log the averages
            for i in range(len(k_train_val_results[0])):
                # Log the averages over k folds for validation
                [avg_validation_writer.add_scalar(key, np.average([k_train_val_results[j][i]['validation'][key]
                                                                   for j in range(len(k_train_val_results))]), i)
                 for key in k_train_val_results[0][i]['validation']]
                # Log the averages over k folds for train
                [avg_train_writer.add_scalar(key, np.average([k_train_val_results[j][i]['train'][key]
                                                              for j in range(len(k_train_val_results))]), i)
                 for key in k_train_val_results[0][i]['train']]
                # Log the averages over k folds for test (should be a giant straight line)
                [avg_test_writer.add_scalar(key, np.average([k_test_results[j][key]
                                                             for j in range(len(k_test_results))]), i)
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
