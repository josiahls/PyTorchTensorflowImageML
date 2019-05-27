import pytest
import torch

from pytorch_tensorflow_image_ml.utils.dataset_mnist_pytorch import DatasetMNIST, DatasetFashionMNIST


def test_pytorch_mnist():
    """
    Test MNIST dataset
    """
    # Do OS file system check
    try:
        DatasetMNIST(n_rows=10, is_testing=True)
    except IOError:
        pytest.fail('Tried opening mnist_test.csv and failed')

    try:
        DatasetMNIST(n_rows=10)
    except IOError:
        pytest.fail('Tried opening mnist_train.csv and failed')

    # Check that the sizes make sense.
    dataset = DatasetMNIST(n_rows=100)
    assert len(dataset) == 100

    # Get the first sample
    assert torch.eq(dataset[0]['x'], 0)
    assert len(dataset[0]['y']) == 784

    # Check that the sizes make sense.
    dataset = DatasetMNIST(n_rows=100, is_testing=True)
    assert len(dataset) == 100

    # Get the first sample
    assert torch.eq(dataset[0]['x'], 2)
    assert len(dataset[0]['y']) == 784


def test_pytorch_fashion_mnist():
    """
    Test Fashion MNIST dataset
    """
    # Do OS file system check
    try:
        DatasetFashionMNIST(n_rows=10, is_testing=True)
    except IOError:
        pytest.fail('Tried opening fashion-mnist_test.csv and failed')

    try:
        DatasetFashionMNIST(n_rows=10)
    except IOError:
        pytest.fail('Tried opening fashion-mnist_train.csv and failed')

    # Check that the sizes make sense.
    dataset = DatasetFashionMNIST(n_rows=100)
    assert len(dataset) == 100

    # Get the first sample
    assert torch.eq(dataset[0]['x'], 2)
    assert len(dataset[0]['y']) == 784

    # Check that the sizes make sense.
    dataset = DatasetFashionMNIST(n_rows=100, is_testing=True)
    assert len(dataset) == 100

    # Get the first sample
    assert torch.eq(dataset[0]['x'], 0)
    assert len(dataset[0]['y']) == 784
