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
from typing import Tuple

from pytorch_tensorflow_image_ml.utils.PytorchSummaryWriter import PyTorchSummaryWriter
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
                     train_writer: PyTorchSummaryWriter = None, validation_writer: PyTorchSummaryWriter = None) -> list:
        """
        Handles dataset splitting and epoch iteration.

        Args:
            validation_writer:
            train_writer:
            validate_train_split:
            epochs:

        Returns: A dictionary containing log entries.

        """
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
            # If the tensorboard writers are available, then write them.
            if train_writer is not None:
                [train_writer.add_scalar(key, scalar_train_results[key], epoch) for key in scalar_train_results]
            if validation_writer is not None:
                [validation_writer.add_scalar(key, scalar_val_results[key], epoch) for key in scalar_val_results]

        # Log data about best and worst samples.
        train_samples, val_samples = self.evaluate_samples(train_dataset, validation_dataset)
        self.write_samples(train_samples, val_samples, train_writer, validation_writer, self.dataset.sample_type,
                           self.dataset.image_shape)

        return scalar_results

    def write_samples(self, train_samples, val_samples, training_writer: PyTorchSummaryWriter,
                      validation_writer: PyTorchSummaryWriter, sample_type,
                      image_shape: Tuple[int, int, int] = None):
        """
        Handles cleanly writing samples to tensorboard.

        Goal is to allow easily logging the best and worst samples no matter the sample type.

        Notes:
            Currently only handles images. Goal is to extend this to text, audio, etc.

        Args:
            val_samples:
            train_samples:
            training_writer:
            validation_writer:
            sample_type (str): Can be 'image'. Future implementation will allow text and audio.
            image_shape ((int, int, int)): If the sample_type is 'image' then an image shape will be required.

        Returns:

        """
        if sample_type == 'image':
            assert image_shape is not None, 'image_shape needs to be defined as (Channel, Height, Width)'

            for key in train_samples:
                for i in count():
                    sample = train_samples[key].get_nowait()
                    image = sample.x.reshape(*image_shape)
                    image_plot = self.image_to_figure_to_tf_image(image, sample.layer_activations,
                                                                  f'\n\n\nActual Y: {sample.y} \n'
                                                                  f'Predicted Y: {sample.pred_y}\n'
                                                                  f'Confidence: {sample.score}')
                    training_writer.add_image(key, image_plot, i, dataformats='HWC')
                    if train_samples[key].empty():
                        break

            if validation_writer is not None:
                for key in val_samples:
                    for i in count():
                        sample = val_samples[key].get_nowait()
                        image = sample.x.reshape(*image_shape)
                        image_plot = self.image_to_figure_to_tf_image(image, sample.layer_activations,
                                                                      f'Actual Y: {sample.y} \n'
                                                                      f'Predicted Y: {sample.pred_y}\n'
                                                                      f'Confidence: {sample.score}')
                        validation_writer.add_image(key, image_plot, i, dataformats='HWC')
                        if val_samples[key].empty():
                            break

    # noinspection PyMethodMayBeStatic,PyUnresolvedReferences
    def image_to_figure_to_tf_image(self, image, layer_activations, text: str):
        """
        Converts an image to a figure with some informative text.

        Then uses PIL to convert the figure to png.

        Finally, we convert the png to a 3 channel numpy image by keeping the first 3 channels.

        Args:
            image:
            text:

        Returns:
        """
        figure = plt.figure(figsize=(5, 10))
        plt.subplot(1 + len(layer_activations), 1, 1)
        plt.title(text)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # Check if the image is grey scale
        if image.shape[0] == 1:
            plt.imshow(image.squeeze(0), cmap=plt.cm.binary)
        else:
            plt.imshow(image)
        for i, layer_activation in enumerate(layer_activations):
            plt.subplot(1 + len(layer_activations), 1, i + 2)
            plt.title(f'Layer {layer_activation}')
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            shape_format = layer_activations[layer_activation].shape
            activation = np.copy(layer_activations[layer_activation])
            square_root = math.sqrt(shape_format[1])
            if square_root.is_integer():
                activation = activation.reshape(int(square_root), int(square_root))
            plt.imshow(activation, cmap=plt.cm.binary)

        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Create Image object
        return np.array(PIL.Image.open(buf))[:, :, :3]

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

        return ({'best_train_samples': best_train_samples, 'worst_train_samples': worst_train_samples},
                {'best_val_samples': best_val_samples, 'worst_val_samples': worst_val_samples})

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
