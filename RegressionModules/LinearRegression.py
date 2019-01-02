import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr



class LinearRegression:

    def __init__(self):
        self.regressor = lr()
        self.was_trained = False
        self.was_tested = False

    def feature_scaling_integers(x_train,
                                 x_test,
                                 features_to_scale: list = None):
        # sc = StandardScaler()
        # FIXME: find a way to scale only specific colums that will be passed in the **args
        # if features_to_scale is not None:
        #     for index in features_to_scale:
        #         x_train[:, index] = sc.fit_transform(x_train[:, index])
        #         x_test[:, index] = sc.transform(x_test[:, index])
        #     return x_train, x_test

        # return sc.fit_transform(x_train), sc.transform(x_test)
        # no need here, the model takes care of it on its own.
        raise NotImplemented("no need in this module")

    def train_module(self,
                     x_train_dataset,
                     y_train_dataset):

        self.regressor.fit(X=x_train_dataset,
                           y=y_train_dataset)
        self.was_trained = True
        return

    def test_module(self,
                    x_test):
        if self.was_trained is False:
            print(' the model must be trained first!')
            raise Exception("entry to only trained modules")
        y_predicted = self.regressor.predict(X=x_test)
        self.was_tested = True
        return y_predicted

    def run_module(self, x_dataset):
        if not self.was_trained:
            print(' the model must be trained first!')
            raise Exception("entry to only trained and tested modules")
        prediction = self.regressor.predict(X=x_dataset)
        return prediction

    def visualize_module(self,
                         X_dataset,
                         Y_dataset,
                         title: str='test title',
                         dots_color: str='red',
                         line_color: str = 'blue',
                         xlabel: str = 'test x label',
                         ylabel: str = 'test y label'):
        plt.scatter(X_dataset, Y_dataset, color=dots_color)
        plt.plot(X_dataset, self.regressor.predict(X_dataset), color=line_color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def __str__(self):
        pass


