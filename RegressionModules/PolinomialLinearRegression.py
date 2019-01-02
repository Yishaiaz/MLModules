import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression as lr
from RegressionModules import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class PolinomialLinearRegression:

    def __init__(self,
                 degree: int=2):
        self.polynomial_regressor = PolynomialFeatures(degree=degree)
        self.linear_regression_module = LinearRegression.LinearRegression()
        self.was_trained = False
        # self.was_tested = False

    def remove_dummy_variables(self,
                               dataset,
                               index_of_column_to_change: int = 0):
        # number_of_different_states = set(item for item in dataset[index_of_column_to_change]).__len__()
        # this module will take care of the dummy variables on its own.
        raise NotImplemented("no need in this module")

    def feature_scaling(self,
                        dataset):
        # there is no need here for feature scaling, this module takes care of that on its own.
        raise NotImplemented("no need in this module")

    def train_module(self,
                     x_dataset,
                     y_dataset):
        x_poly = self.polynomial_regressor.fit_transform(X=x_dataset)
        self.linear_regression_module.train_module(x_poly, y_dataset)
        self.was_trained = True
        return

    def test_module(self,
                    x_test):
        raise NotImplemented("no need in this module")
        # if self.was_trained is False:
        #     print(' the model must be trained first!')
        #     raise Exception("entry to only trained modules")
        # y_pred = self.regressor.predict(X=x_test)
        # self.was_tested = True
        # return y_pred

    def run_module(self, x_dataset):
        if not self.was_trained:
            raise NotImplemented("train the module first!")
        return self.linear_regression_module.run_module(self.polynomial_regressor.fit_transform(x_dataset))



    def visualize_module(self,
                         X_dataset,
                         Y_dataset,
                         title: str = 'test title',
                         dots_color: str = 'red',
                         line_color: str = 'blue',
                         xlabel: str = 'test x label',
                         ylabel: str = 'test y label'):
        plt.scatter(X_dataset, Y_dataset, color=dots_color)
        plt.plot(X_dataset,
                 self.linear_regression_module.run_module(self.polynomial_regressor.fit_transform(X_dataset)),
                 color=line_color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def __str__(self):
        return f"the multiple linear regression module, trained={self.was_trained}, tested{self.was_tested}"
