import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as cm
from matplotlib.colors import ListedColormap


class LogisticRegressionModule:

    def __init__(self):
        self.LRM_classifier = LogisticRegression(random_state=0)
        self.was_trained = False
        self.was_scaled = False
        self.was_tested = False
        self.sc_x = None
        # self.was_tested = False

    def remove_dummy_variables(self,
                               dataset,
                               index_of_column_to_change: int = 0):
        # number_of_different_states = set(item for item in dataset[index_of_column_to_change]).__len__()
        # this module will take care of the dummy variables on its own.
        raise NotImplemented("no need in this module")

    def feature_scaling(self,
                        x_train,
                        x_test) ->tuple:
        self.sc_x = StandardScaler()
        x_train = self.sc_x.fit_transform(x_train)
        x_test = self.sc_x.transform(x_test)
        self.was_scaled = True
        return x_train, x_test
        # raise NotImplemented("no need in this module")

    def feature_scaling_for_new_entries(self,
                                        dataset):
        if not self.was_scaled:
            raise Warning('the original training data set must be scaled with the same SVR class BEFORE scaling the target dataset')
        return self.sc_x.transform(dataset)
        # raise NotImplemented("no need in this module")

    def train_module(self,
                     x_dataset,
                     y_dataset):
        self.LRM_classifier.fit(X=x_dataset, y=y_dataset)
        self.was_trained = True
        return

    def test_module(self,
                    x_test):
        # raise NotImplemented("no need in this module")
        if self.was_trained is False:
            raise Exception("entry to only trained modules")
        y_pred = self.LRM_classifier.predict(X=x_test)
        self.was_tested = True
        return y_pred

    def create_confusion_matrix(self,
                                y_true,
                                y_pred):
        return cm(y_true=y_true, y_pred=y_pred)

    def run_module(self, x_dataset):
        if not self.was_trained:
            raise NotImplemented("train the module first!")
        return self.LRM_classifier.predict(X=x_dataset)

    def visualize_module(self,
                         X_dataset,
                         Y_dataset,
                         title: str = 'test title',
                         dots_color: str = 'red',
                         line_color: str = 'blue',
                         xlabel: str = 'test x label',
                         ylabel: str = 'test y label',
                         resolution: float = 0.1):
        x1, x2 = np.meshgrid(np.arange(start=X_dataset[:, 0].min()-1, stop=X_dataset[:, 0].max()+1, step=0.01),
                             np.arange(start=X_dataset[:, 1].min() - 1, stop=X_dataset[:, 1].max() + 1, step=0.01))
        plt.contourf(x1,
                     x2,
                     self.LRM_classifier.predict(np.array([x1.ravel(),
                                                           x2.ravel()]).T).reshape(x1.shape),
                     alpha=0.75,
                     cmap=ListedColormap(('red', 'green')))
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        for i, j in enumerate(np.unique(Y_dataset)):
            print(f"{j}, {i}")
            plt.scatter(X_dataset[Y_dataset[:, 0] == j, 0],
                        X_dataset[Y_dataset[:, 0] == j, 1],
                        c=ListedColormap(('red', 'green'))(i),
                        label=j)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def __str__(self):
        return f"the multiple linear regression module, trained={self.was_trained}, tested{self.was_tested}"
