#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import argparse


class Config:
    def __init__(self):

        self.parser = argparse.ArgumentParser()
        # Setup information (most important)
        self.agents = None  # type: str

        """ Training Config """
        self.split_percent = 0.7
        self.validate_train_split = True
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.epochs = 20
        self.batch_size = 10
        self.k_folds = 1

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])

        self.parser = None
