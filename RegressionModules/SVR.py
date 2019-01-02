import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression as lr
from RegressionModules import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class SVRModule:

    def __init__(self,
                 kernel = 'rbf',
                 degree: int=2):
        self.SVR_regressor = SVR(kernel=kernel)
        # self.linear_regression_module = LinearRegression.LinearRegression()
        self.was_trained = False
        self.was_scaled = False
        self.sc_x=None
        self.sc_y=None
        # self.was_tested = False

    def remove_dummy_variables(self,
                               dataset,
                               index_of_column_to_change: int = 0):
        # number_of_different_states = set(item for item in dataset[index_of_column_to_change]).__len__()
        # this module will take care of the dummy variables on its own.
        raise NotImplemented("no need in this module")

    def feature_scaling(self,
                        x_dataset,
                        y_dataset) ->tuple:
        self.sc_x = StandardScaler()
        self.sc_y = StandardScaler()
        x_dataset = self.sc_x.fit_transform(x_dataset)
        y_dataset = self.sc_y.fit_transform(y_dataset)
        self.was_scaled = True
        return x_dataset, y_dataset

    def feature_scaling_for_new_entries(self,
                                        dataset,
                                        is_x=True):
        if not self.was_scaled:
            raise Warning('the original training data set must be scaled with the same SVR class BEFORE scaling the target dataset')
            return None
        if is_x:
            return self.sc_x.transform(dataset)
        return self.sc_y.transform(dataset)

    def train_module(self,
                     x_dataset,
                     y_dataset):
        self.SVR_regressor.fit(X=x_dataset,y=y_dataset)
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
        return self.sc_y.inverse_transform(self.SVR_regressor.predict(x_dataset))

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
                 self.SVR_regressor.predict(X_dataset),
                 color=line_color)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def __str__(self):
        return f"the multiple linear regression module, trained={self.was_trained}, tested{self.was_tested}"
