import torch
import numpy as np
from typing import Dict


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    # noinspection PyUnresolvedReferences
    def __call__(self, sample: dict):
        return {c: torch.tensor(sample[c]).float() for c in sample}


class TreatCategorical(object):
    """ Makes sure the 'y' tensor is a long. """

    # noinspection PyUnresolvedReferences
    def __call__(self, sample: dict):
        return {'x': sample['x'], 'y': sample['y'].long()}


class ToOneHot(object):
    """ Converts y to a One Hot vector """
    def __init__(self, n_classes=1):
        self.n_classes = n_classes
        self.one_hot_y = torch.LongTensor(1, n_classes).zero_()

        # noinspection PyUnresolvedReferences
    def __call__(self, sample: dict):
        self.one_hot_y.zero_()
        return {'x': sample['x'], 'y': self.one_hot_y.scatter_(1, sample['y'].unsqueeze(0), 1).squeeze(0).long()}


class MNISTToXY(object):
    """
    Converts a sample to its x, y. Complexity is how we define what is a single image and which is not.
    The key '5' is the label, and everything else is the image that has been flattened.

    The y labels are scalars, so we need to add a dimension to them.
    """

    # noinspection PyUnresolvedReferences
    def __call__(self, sample: dict) -> Dict[str, torch.tensor]:
        return {'x': list(sample.values())[1:], 'y': np.expand_dims(list(sample.values())[0], 0)}
