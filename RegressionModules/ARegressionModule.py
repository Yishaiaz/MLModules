from abc import ABC, abstractmethod

import RegressionModules.LinearRegression


class ARegressionModule(ABC):

    @abstractmethod
    def train_module(self):
        pass

    @abstractmethod
    def test_module(self):
        pass

    @abstractmethod
    def run_module(self):
        pass

    @abstractmethod
    def __str__(self):
        return f"The {type(self)} class, receives a preprocessed dataset and returns the "


ARegressionModule.register(RegressionModules.LinearRegression)
