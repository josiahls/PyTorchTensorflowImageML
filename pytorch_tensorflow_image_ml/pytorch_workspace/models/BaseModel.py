import copy
import queue
from collections import deque

import PIL
import io
import math
from abc import ABC
from itertools import count
from queue import Queue, PriorityQueue
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import numpy as np
import torchvision
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List

from pytorch_tensorflow_image_ml.utils.callbacks import Callback
from pytorch_tensorflow_image_ml.utils.pytorch_summary_writer import PyTorchSummaryWriter
from pytorch_tensorflow_image_ml.utils.config import Config
from pytorch_tensorflow_image_ml.utils.datasets_pytorch import BasePyTorchDataset
from pytorch_tensorflow_image_ml.utils.sample_object import SampleObject


class BaseModel(ABC):
    NAME = ''

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
        self.activations = {}

    def forward(self, input_tensor):
        return self.model.forward(input_tensor)

    def get_activation(self, name):
        def hook(_, __, output):
            self.activations[name] = output.detach()

        return hook

    def run_n_epochs(self, epochs: int, validate_train_split: bool = False,
                     callbacks: List[type(Callback)]=None) -> Tuple[list, dict]:
        """
        Handles dataset splitting and epoch iteration.

        Args:
            callbacks:
            validation_writer:
            train_writer:
            validate_train_split:
            epochs:

        Returns: A dictionary containing log entries.

        """
        callbacks = callbacks if callbacks else None
        train_dataset = self.dataset
        val_dataset_loader = None
        validation_dataset = None
        scalar_results = []

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
            scalar_train_results, scalar_val_results = self.run_epoch(scalar_results, train_dataset_loader,
                                                                      val_dataset_loader)
            for callback in callbacks:
                callback.on_epoch_end(scalar_train_results=scalar_train_results, scalar_val_results=scalar_val_results,
                                      epoch=epoch)

        # Log data about best and worst samples.
        train_samples, val_samples = self.evaluate_samples(train_dataset, validation_dataset)

        non_linear_results = {'train_samples': copy.deepcopy(train_samples),
                              'validation_samples': copy.deepcopy(val_samples),
                              'sample_type': self.dataset.sample_type,
                              'image_shape': self.dataset.image_shape}

        return scalar_results, non_linear_results

    def evaluate_samples(self, train_dataset, val_dataset, buffer_size=10):
        """
        Takes the train and validation datasets and gets the best and worst performing samples based on the model's
        probabilities.

        Args:
            buffer_size: How many best, worst samples do we want to keep?
            train_dataset:
            val_dataset:

        Returns:

        """

        # Priority Queues act as a buffer. + 1 is added since 1 extra sample will always be removed on last iter
        best_train_samples = PriorityQueue(buffer_size + 1)
        best_val_samples = PriorityQueue(buffer_size + 1)
        worst_train_samples = PriorityQueue(buffer_size + 1)
        worst_val_samples = PriorityQueue(buffer_size + 1)

        # Iterate through the train samples
        for element in train_dataset:
            pred_y = self.forward(element['x'].unsqueeze(0))
            best_train_samples.put_nowait(SampleObject(element['x'], element['y'], pred_y.argmax(),
                                                       pred_y[0, pred_y.argmax()].detach().numpy(),
                                                       self.activations.copy()))
            worst_train_samples.put_nowait(SampleObject(element['x'], element['y'], pred_y.argmax(),
                                                        pred_y[0, pred_y.argmax()].detach().numpy(),
                                                        self.activations.copy(),
                                                        reversed_order=True))
            if best_train_samples.full():
                best_train_samples.get_nowait()
            if worst_train_samples.full():
                worst_train_samples.get_nowait()

        # Iterate through the validation samples
        if val_dataset:
            for element in val_dataset:
                pred_y = self.forward(element['x'].unsqueeze(0))
                best_val_samples.put(SampleObject(element['x'], element['y'], pred_y.argmax(),
                                                  pred_y[0, pred_y.argmax()].detach().numpy(),
                                                  self.activations.copy()))
                worst_val_samples.put(SampleObject(element['x'], element['y'], pred_y.argmax(),
                                                   pred_y[0, pred_y.argmax()].detach().numpy(),
                                                   self.activations.copy(),
                                                   reversed_order=True))
                if best_val_samples.full():
                    best_val_samples.get_nowait()
                if worst_val_samples.full():
                    worst_val_samples.get_nowait()

        # noinspection PyUnresolvedReferences
        return ({'best_train_samples': best_train_samples.queue,
                 'worst_train_samples': worst_train_samples.queue},
                {'best_val_samples': best_val_samples.queue,
                 'worst_val_samples': worst_val_samples.queue})

    def run_epoch(self, scalar_results, train_dataset_loader, val_dataset_loader):
        """
        Runs a single epoch on the model.

        Args:
            scalar_results:
            train_dataset_loader:
            val_dataset_loader:

        Returns:

        """
        scalar_train_results = self.step(train_dataset_loader)
        scalar_val_results = None

        if val_dataset_loader is not None:
            scalar_val_results = self.step(val_dataset_loader, evaluate=True)

        scalar_results.append({'train': scalar_train_results, 'validation': scalar_val_results})
        return scalar_train_results, scalar_val_results

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
        scalar_results = {'loss': np.average(saved_loss),
                          'f1_score': f1_score(saved_y, saved_pred_y.argmax(1), average='weighted'),
                          'accuracy': accuracy_score(saved_y, saved_pred_y.argmax(1)),
                          'precision': precision_score(saved_y, saved_pred_y.argmax(1), average='weighted'),
                          'recall': recall_score(saved_y, saved_pred_y.argmax(1), average='weighted')}
        return scalar_results

    def get_layer_activations(self):
        """
        Should return a dict of layer activations.

        Returns:

        """
        return None
