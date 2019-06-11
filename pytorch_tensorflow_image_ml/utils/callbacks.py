import io
import math
from abc import ABC
from collections import deque
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pytorch_tensorflow_image_ml.utils.pytorch_summary_writer import PyTorchSummaryWriter


class Callback(ABC):
    def __init__(self, model_name, dataset_name, k=-1):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.k = k

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_step_end(self,  **kwargs):
        pass

    def on_epoch_begin(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_k_train_end(self, **kwargs):
        pass


class TensorboardCallback(Callback):
    def __init__(self, model_name, dataset_name, k, writer_prefix):
        super().__init__(model_name, dataset_name, k)
        self.w_p = writer_prefix
        self.w_post = f'_{self.model_name}_{self.dataset_name}'
        self.w_post = self.w_post if self.k == -1 else self.w_post + f'_k_{self.k}'
        self.train_writer = PyTorchSummaryWriter(f'{self.w_p}_train{self.w_post}')
        self.validation_writer = PyTorchSummaryWriter(f'{self.w_p}_validation{self.w_post}')

    def on_epoch_end(self, scalar_train_results, scalar_val_results, epoch, **kwargs):
        [self.train_writer.add_scalar(key, scalar_train_results[key], epoch) for key in scalar_train_results]
        [self.validation_writer.add_scalar(key, scalar_val_results[key], epoch) for key in scalar_val_results]

    def on_train_end(self, train_val_results, test_results, non_linear_results, **kwargs):
        sample_type = non_linear_results['sample_type']
        image_shape = non_linear_results['image_shape']
        train_samples = non_linear_results['train_samples']
        val_samples = non_linear_results['validation_samples']

        if sample_type == 'image':
            assert image_shape is not None, 'image_shape needs to be defined as ' \
                                                               '(Channel, Height, Width)'

            for key in train_samples:
                temp_queue = deque(train_samples[key])
                for i in count():
                    sample = temp_queue.pop()
                    image = sample.x.reshape(*image_shape)
                    image_plot = self.image_to_figure_to_tf_image(image, sample.layer_activations,
                                                                  f'\n\n\nActual Y: {sample.y} \n'
                                                                  f'Predicted Y: {sample.pred_y}\n'
                                                                  f'Confidence: {sample.score}')
                    self.train_writer.add_image(key, image_plot, i, dataformats='HWC')
                    if not temp_queue:
                        break

            for key in val_samples:
                temp_queue = deque(val_samples[key])
                for i in count():
                    sample = temp_queue.pop()
                    image = sample.x.reshape(*image_shape)
                    image_plot = self.image_to_figure_to_tf_image(image, sample.layer_activations,
                                                                  f'Actual Y: {sample.y} \n'
                                                                  f'Predicted Y: {sample.pred_y}\n'
                                                                  f'Confidence: {sample.score}')
                    self.validation_writer.add_image(key, image_plot, i, dataformats='HWC')
                    if not temp_queue:
                        break
        self.train_writer.close()
        self.validation_writer.close()


    def image_to_figure_to_tf_image(self, image, layer_activations, text: str):
        """
        Converts an image to a figure with some informative text.

        Then uses PIL to convert the figure to png.

        Finally, we convert the png to a 3 channel numpy image by keeping the first 3 channels.

        Args:
            layer_activations:
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
        return np.array(Image.open(buf))[:, :, :3]

    def on_k_train_end(self, k_train_val_results, k_test_results, **kwargs):
        avg_test_writer = PyTorchSummaryWriter(f'{self.w_p}_test{self.w_post}_averaged')
        avg_train_writer = PyTorchSummaryWriter(f'{self.w_p}_train{self.w_post}_averaged')
        avg_validation_writer = PyTorchSummaryWriter(f'{self.w_p}_validation{self.w_post}_averaged')

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