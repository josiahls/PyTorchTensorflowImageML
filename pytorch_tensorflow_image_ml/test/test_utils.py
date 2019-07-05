import pytest
from pytorch_tensorflow_image_ml.utils.file_handling import get_absolute_path
import pandas as pd
import os


def test_absolute_path():
    # Can it find the data directory?
    found_path = get_absolute_path('data')
    assert found_path.__contains__('data'), 'Failed in find the data directory.'
    # Does it throw an error correctly?
    with pytest.raises(IOError):
        get_absolute_path('billy')
    # Does it find sub directories?
    found_path = get_absolute_path('mnist')
    assert found_path.__contains__('mnist'), 'Failed in find the mnist directory.' \
                                             'As a note, this directory is expecting ' \
                                             ' the mnist dataset to exist for testing purposes.'
    # Does it find sub directories?
    found_path = get_absolute_path(['test', 'runs'])
    assert found_path.__contains__('runs'), 'Failed in find the runs directory.' \
                                            'As a note, this directory is expecting ' \
                                            ' the mnist dataset to exist for testing purposes.'


def test_pandas_dataset_load():
    absolute_path_to_mnist = get_absolute_path('mnist')

    # The os.path.joint is better than using 'some_path' + '/' + 'filename' because it will determine the separator '/'
    # as needing to either be '/' or '\'. This is a safer approach to combining paths.
    df = pd.read_csv(os.path.join(absolute_path_to_mnist, 'mnist_train.csv'), nrows=5)
    print('here as a place holder')
