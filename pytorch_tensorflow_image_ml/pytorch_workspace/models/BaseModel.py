from abc import ABC

import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.datasets_pytorch import BasePyTorchDataset


class BaseModel(ABC):
    def __init__(self, config: Config, dataset: BasePyTorchDataset):
        """
        Base class for models. Contains an interface / implementation for forward,
        epoch iteration, and single epoch step. Also binds together data for logging.

        Args:
            config: Configuration object for the model
            dataset: The dataset to train on.
        """
        self.config = config
        self.dataset = dataset
        # if gpu is to be used
        # noinspection PyUnresolvedReferences
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.criterion = None  # type: nn.CrossEntropyLoss
        self.optimizer = None  # type: optim
        self.model = None  # type: nn.Module

    def forward(self, input_tensor):
        return self.model.forward(input_tensor)

    def run_n_epochs(self, epochs: int, validate_train_split: bool = False) -> list:
        """
        Handles dataset splitting and epoch iteration.

        Args:
            validate_train_split:
            epochs:

        Returns: A dictionary containing log entries.

        """
        train_dataset = self.dataset
        val_dataset_loader = None
        results = []

        # If we want validation set data, then break the training dataset into 2 smaller datasets.
        if validate_train_split:
            lengths = (int(len(self.dataset) * self.config.split_percent),
                       int(len(self.dataset) * (1 - self.config.split_percent)))

            # noinspection PyUnresolvedReferences
            train_dataset, validation_dataset = torch.utils.data.random_split(self.dataset, lengths)
            val_dataset_loader = DataLoader(dataset=validation_dataset, batch_size=self.config.batch_size, shuffle=True)

        # Init the DataLoaders
        train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=self.config.batch_size, shuffle=True)

        for epoch in range(epochs):
            train_results = self.step(train_dataset_loader)
            val_results = None

            if validate_train_split:
                val_results = self.step(val_dataset_loader, evaluate=True)

            results.append({'train': train_results, 'validation': val_results})

        return results

    def step(self, data_loader, evaluate=False):
        """
        Per epoch, we need to:
        - Iterate through the data loader
        - Get the loss
        - Back prop

        Args:
            data_loader: The Data Loader to train on and/or evaluate on.
            evaluate: Whether to do just model evaluation without training.

        Returns:

        """
        self.model.train() if not evaluate else self.model.eval()
        saved_y = None
        saved_pred_y = None
        saved_loss = []

        for i, data in enumerate(data_loader):
            x, y = (data['x'], data['y'])

            # Zero out the gradients
            self.optimizer.zero_grad()

            pred_y = self.model.forward(x)

            # forward + backward + optimize
            loss = self.criterion(pred_y, y)

            if not evaluate:
                loss.backward()
                self.optimizer.step()

            saved_y = y if saved_y is None else torch.cat((saved_y, y))
            saved_pred_y = pred_y if saved_pred_y is None else torch.cat((saved_pred_y, pred_y))
            saved_loss.append(loss.detach().cpu())

        saved_y = saved_y.detach()
        saved_pred_y = saved_pred_y.detach()
        results = {'loss': np.average(saved_loss),
                   'f1_score': f1_score(saved_y, saved_pred_y.argmax(1), average='weighted'),
                   'accuracy': accuracy_score(saved_y, saved_pred_y.argmax(1)),
                   'precision': precision_score(saved_y, saved_pred_y.argmax(1), average='weighted'),
                   'recall': recall_score(saved_y, saved_pred_y.argmax(1), average='weighted')}
        return results
