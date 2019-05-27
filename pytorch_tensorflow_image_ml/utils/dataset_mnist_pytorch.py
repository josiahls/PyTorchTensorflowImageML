import os

from torch.distributions import Transform
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import Compose

from pytorch_tensorflow_image_ml.utils.file_handling import get_absolute_path
from pytorch_tensorflow_image_ml.utils.transforms import MNISTToXY, ToTensor


class DatasetMNIST(Dataset):
    def __init__(self, data_dir='data', folder_dir='mnist', transform: Transform = Compose([MNISTToXY(), ToTensor()]),
                 n_rows=None, is_testing=False):
        """
        Loads MNIST dataset using PyTorch API based dataset objects.

        A core methodology in terms of dataset building will be using a
        test directory and a train directory.

        Then the train directory can be shuffled into validation and train sets.

        See Also:
            https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
            https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6?gi=36f25b78c7ef

        """
        self.transform = transform
        self.n_rows = n_rows
        self.data_dir = data_dir
        self.folder_dir = folder_dir
        self.dataset_root_path = get_absolute_path(folder_dir)

        self.csv_name = 'mnist_test.csv' if is_testing else 'mnist_train.csv'
        self.main_dataframe = pd.read_csv(os.path.join(self.dataset_root_path, self.csv_name), nrows=self.n_rows)

    def __getitem__(self, index):
        sample = self.main_dataframe.iloc[int(index)].to_dict()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.main_dataframe)


class DatasetFashionMNIST(Dataset):
    def __init__(self, data_dir='data', folder_dir='fashionmnist', transform: Transform = Compose([MNISTToXY(), ToTensor()]),
                 n_rows=None, is_testing=False):
        """
        Loads Fashion MNIST dataset using PyTorch API based dataset objects.

        A core methodology in terms of dataset building will be using a
        test directory and a train directory.

        Then the train directory can be shuffled into validation and train sets.

        See Also:
            https://www.kaggle.com/zalando-research/fashionmnist
            https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
            https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6?gi=36f25b78c7ef

        """
        self.transform = transform
        self.n_rows = n_rows
        self.data_dir = data_dir
        self.folder_dir = folder_dir
        self.dataset_root_path = get_absolute_path(folder_dir)

        self.csv_name = 'fashion-mnist_test.csv' if is_testing else 'fashion-mnist_train.csv'
        self.main_dataframe = pd.read_csv(os.path.join(self.dataset_root_path, self.csv_name), nrows=self.n_rows)

    def __getitem__(self, index):
        sample = self.main_dataframe.iloc[int(index)].to_dict()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.main_dataframe)

