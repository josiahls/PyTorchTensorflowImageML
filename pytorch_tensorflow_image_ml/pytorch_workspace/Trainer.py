from typing import List
from torch.utils.data import DataLoader
from pytorch_tensorflow_image_ml.pytorch_workspace.models.BaseModel import BaseModel
from pytorch_tensorflow_image_ml.utils.callbacks import Callback
from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.datasets_pytorch import BasePyTorchDataset


class Trainer(object):

    def __init__(self, config: Config, models=None, datasets: List[BasePyTorchDataset] = None,
                 callbacks: List[type(Callback)]=None):
        """
        Handles model iteration and datasets iteration.
        Allows for a quick convenient way to test many models.

        Args:
            callbacks:
            Useful for differentiating unit testing runs from actual runs.
            config:
            models:
            datasets:
        """
        self.callbacks = callbacks if callbacks else None
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

            model_instance = model(self.config, dataset)

            callbacks = [callback(model_instance.NAME, dataset.name, k) for callback in self.callbacks]
            for callback in callbacks:
                callback.on_train_begin(model=model_instance)

            train_val_results, non_linear_results = model_instance.run_n_epochs(self.config.epochs,
                                                                                self.config.validate_train_split,
                                                                                callbacks=callbacks)
            dataset.toggle_test_train(True)
            test_results = model_instance.step(DataLoader(dataset=dataset,
                                                          batch_size=self.config.batch_size, shuffle=True), True)
            dataset.toggle_test_train(False)

            k_train_val_results.append(train_val_results)
            k_test_results.append(test_results)

            for callback in callbacks:
                callback.on_train_end(train_val_results=train_val_results, test_results=test_results,
                                      non_linear_results=non_linear_results)

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
            k_train_val_results = []
            k_test_results = []

            callbacks = [callback(model.NAME, dataset.name) for callback in self.callbacks]

            self.run_model_on_dataset(model, dataset, k_train_val_results, k_test_results)

            for callback in callbacks:
                callback.on_k_train_end(k_train_val_results=k_train_val_results, k_test_results=k_test_results)

    def run_models_on_datasets(self):
        """
        Primarily iterates through the `self.models` field, and tests each model
        on the list of datasets.

        Returns:

        """
        for model in self.models:
            self.run_model_on_datasets(model, self.datasets)
