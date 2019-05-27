import torch
import numpy as np
from typing import Dict


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    # noinspection PyUnresolvedReferences
    def __call__(self, sample: dict):
        return {c: torch.tensor(sample[c]).float() for c in sample}


class MNISTToXY(object):
    """
    Converts a sample to its x, y. Complexity is how we define what is a single image and which is not.
    The key '5' is the label, and everything else is the image that has been flattened.

    The y labels are scalars, so we need to add a dimension to them.
    """
    # noinspection PyUnresolvedReferences
    def __call__(self, sample: dict) -> Dict[str, torch.tensor]:
        return {'y': np.expand_dims(list(sample.values())[0], 0), 'x': list(sample.values())[1:]}
