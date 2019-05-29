import pytest
import torch
from torchvision.transforms import Compose

from pytorch_tensorflow_image_ml.utils.datasets_pytorch import DatasetMNIST, DatasetFashionMNIST
from pytorch_tensorflow_image_ml.utils.transforms import MNISTToXY, ToTensor, TreatCategorical, ToOneHot


def test_mnist_pytorch():
    """
    Test MNIST dataset
    """
    # Do OS file system check
    try:
        DatasetMNIST(n_rows=10, is_testing=True, transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]),
                     categorical=False)
    except IOError:
        pytest.fail('Tried opening mnist_test.csv and failed')

    try:
        DatasetMNIST(n_rows=10, transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]), categorical=False)
    except IOError:
        pytest.fail('Tried opening mnist_train.csv and failed')

    # Check that the sizes make sense.
    dataset = DatasetMNIST(n_rows=100, transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]), categorical=False)
    assert len(dataset) == 100

    # Get the first sample
    assert torch.eq(dataset[0]['y'], 0)
    assert len(dataset[0]['x']) == 784

    # Check that the sizes make sense.
    dataset = DatasetMNIST(n_rows=100, is_testing=True,
                           transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]), categorical=False)
    assert len(dataset) == 100

    # Get the first sample
    assert torch.eq(dataset[0]['y'], 2)
    assert len(dataset[0]['x']) == 784


def test_mnist_categorical_pytorch():
    """
    Test MNIST dataset
    """
    # Do OS file system check
    try:
        DatasetMNIST(n_rows=10, is_testing=True, one_hot=True,
                     transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]))
    except IOError:
        pytest.fail('Tried opening mnist_test.csv and failed')

    try:
        DatasetMNIST(n_rows=10, one_hot=True, transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]))
    except IOError:
        pytest.fail('Tried opening mnist_train.csv and failed')

    # Check that the sizes make sense.
    dataset = DatasetMNIST(n_rows=100, one_hot=True, transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]))
    assert len(dataset) == 100

    # Get the first sample
    assert torch.eq(dataset[0]['y'].argmax(), 0)
    assert len(dataset[0]['x']) == 784

    # Check that the sizes make sense.
    dataset = DatasetMNIST(n_rows=100, is_testing=True, one_hot=True,
                           transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]))
    assert len(dataset) == 100

    # Get the first sample
    assert torch.eq(dataset[0]['y'].argmax(), 2)
    assert len(dataset[0]['x']) == 784


def test_fashion_mnist_pytorch():
    """
    Test Fashion MNIST dataset
    """
    # Do OS file system check
    try:
        DatasetFashionMNIST(n_rows=10, is_testing=True, categorical=False,
                            transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]))
    except IOError:
        pytest.fail('Tried opening fashion-mnist_test.csv and failed')

    try:
        DatasetFashionMNIST(n_rows=10, categorical=False,
                            transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]))
    except IOError:
        pytest.fail('Tried opening fashion-mnist_train.csv and failed')

    # Check that the sizes make sense.
    dataset = DatasetFashionMNIST(n_rows=100, categorical=False,
                                  transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]))
    assert len(dataset) == 100

    # Get the first sample
    assert torch.eq(dataset[0]['y'], 2)
    assert len(dataset[0]['x']) == 784

    # Check that the sizes make sense.
    dataset = DatasetFashionMNIST(n_rows=100, is_testing=True, categorical=False,
                                  transform=Compose([MNISTToXY(), ToTensor(), TreatCategorical()]))
    assert len(dataset) == 100

    # Get the first sample
    assert torch.eq(dataset[0]['y'], 0)
    assert len(dataset[0]['x']) == 784
