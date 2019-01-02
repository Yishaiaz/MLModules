import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression as lr


class MultipleLineaerRegression:

    def __init__(self):
        self.regressor = lr()
        self.was_trained = False
        self.was_tested = False

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

    def variable_elimination(self,
                             strategy: str = "backward",
                             X_dataset=None,
                             y_dataset=None,
                             sl=0.05):
        # backward/forward/bi-directional/all-possible considering R squared is defaultive.
        def backward_elimination(x_dataset,
                                 sl=0.05,
                                 strategy: str = "rsquared"):

            def backward_elimination_with_r_squared(x_dataset,
                                                    sl=0.05):
                numVars = len(x_dataset[0])
                temp = np.zeros((50, 6)).astype(int)
                for i in range(0, numVars):
                    regressor_OLS = sm.OLS(endog=y_dataset,
                                           exog=x_dataset).fit()
                    maxVar = max(regressor_OLS.pvalues).astype(float)
                    adjR_before = regressor_OLS.rsquared_adj.astype(float)
                    if maxVar > sl:
                        for j in range(0, numVars - i):
                            if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                                temp[:, j] = x_dataset[:, j]
                                x_dataset = np.delete(x_dataset, j, 1)
                                tmp_regressor = sm.OLS(endog=y_dataset, exog=x_dataset).fit()
                                adjR_after = tmp_regressor.rsquared_adj.astype(float)
                                if (adjR_before >= adjR_after):
                                    x_rollback = np.hstack((x_dataset, temp[:, [0, j]]))
                                    x_rollback = np.delete(x_rollback, j, 1)
                                    # print(regressor_OLS.summary())
                                    return x_rollback
                                else:
                                    continue
                regressor_OLS.summary()
                return x_dataset

            def backward_elimination_only_pvalue(x_dataset,
                                                 sl=0.05):
                X_opt = x_dataset[:, :]
                while True:
                    # fitting the new optimal dataset
                    regressor_OLS = sm.OLS(endog=y_dataset,
                                           exog=X_opt).fit()
                    # finding the current lowest p value variable
                    print(regressor_OLS.summary())
                    index = 0
                    highest_pvalue = 0
                    highest_value_index = None
                    for pvalue_of_variable in regressor_OLS.summary().tables[1]:
                        if index > 0:
                            if highest_pvalue < float(pvalue_of_variable[4].data):
                                highest_pvalue = float(pvalue_of_variable[4].data)
                                highest_value_index = index - 1
                        index += 1
                    if highest_pvalue > sl:
                        X_opt = np.delete(arr=X_opt,
                                          obj=highest_value_index,
                                          axis=1)
                    else:
                        return X_opt
            OPTIONAL_ELIMINATION_TYPE = {
                'rsquared': backward_elimination_with_r_squared,
                'pvalue': backward_elimination_only_pvalue
            }
            func_to_activate = OPTIONAL_ELIMINATION_TYPE.get(strategy, "not a implemented strategy")
            return func_to_activate(x_dataset=x_dataset, sl=sl)

        def forward_elimination(x_dataset,
                                sl=0.05):
            raise NotImplemented

        def bi_directional_elimination(x_dataset,
                                       sl=0.05):
            raise NotImplemented

        def all_possible_elimination(x_dataset,
                                     sl=0.05):
            raise NotImplemented

        OPTIONAL_ELIMINATION_TYPE = {
            'backward': backward_elimination,
            'forward': forward_elimination,
            'bi-directional': bi_directional_elimination,
            'all-possible': all_possible_elimination
        }
        # removing one of the dummy variables (the first column)
        X_dataset = X_dataset[:, 1:]
        # adding the constant variable column
        X_dataset = np.append(arr=np.ones((X_dataset.__len__(), 1)).astype(int),
                              values=X_dataset,
                              axis=1)
        func_to_activate = OPTIONAL_ELIMINATION_TYPE.get(strategy, lambda: "not a implemented strategy")
        return func_to_activate(x_dataset=X_dataset,
                                sl=0.05,
                                strategy="rsquared")

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
        y_pred = self.regressor.predict(X=x_test)
        self.was_tested = True
        return y_pred

    def run_module(self, x_dataset):
        if not (self.was_trained and self.was_tested):
            print(' the model must be trained first!')
            raise Exception("entry to only trained and tested modules")
        return self.regressor.predict(x_dataset)

    def visualize_module(self,
                         X_train=[[]],
                         Y_train=[],
                         title: str = 'test title',
                         dots_color: str = 'red',
                         line_color: str = 'blue',
                         xlabel: str = 'test x label',
                         ylabel: str = 'test y label'):
        # plt.scatter(X_train, Y_train, color=dots_color)
        # plt.plot(X_train, self.regressor.predict(X_train), color=line_color)
        # plt.title(title)
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt.show()
        raise NotImplemented

    def __str__(self):
        return f"the multiple linear regression module, trained={self.was_trained}, tested{self.was_tested}"
