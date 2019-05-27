from torch.utils.data import Dataset

from pytorch_tensorflow_image_ml.utils.config import Config


class Trainer(object):

    def __init__(self, config: Config, models, datasets: Dataset):
        self.config = config
        self.models = models
        self.datasets = datasets

    def run_model_on_datasets(self):
        pass

    def run_models_on_datasets(self):
        pass
