import pytest
from pytorch_tensorflow_image_ml.utils.file_handling import get_absolute_path


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
