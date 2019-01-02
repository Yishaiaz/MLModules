import testing_and_trying as tt
import PreprocessingModule as ppM
import pandas as pd

if __name__ == '__main__':
    # y = tt.encoding_categorical_variables_by_column(dataset=["yes", "no", "yes", "yes", "no"],
    #                                                 is_results=True,
    #                                                 strategy="encoding")
    # print(y)
    x, y = ppM.importing_the_dataset('DATA_FILES/Machine Learning A-Z/Part 1 - Data Preprocessing/Data.csv')
    x = ppM.fill_up_missing_data(dataset_x=x,
                                 dataset_x_start_of_missing_data_index=1,
                                 dataset_x_end_of_missing_data_index=3,
                                 strategy="mean")
    x = ppM.encoding_categorical_variables_by_column(dataset=x,
                                                     categorical_columns=[0])
    y = ppM.encoding_categorical_variables_by_column(dataset=y,
                                                     categorical_columns=[0],
                                                     strategy="encoding",
                                                     is_results=True)
    x_train, x_test, y_train, y_test = ppM.split_to_train_and_test(x,
                                                                   y,
                                                                   random_state=0,
                                                                   test_size=0.2)
    x_train, x_test = ppM.feature_scaling_integers(x_train, x_test)
    y_train, y_test = ppM.feature_scaling_integers(y_train, y_test)
    print(x_train, x_test, y_train, y_test)
    print(x)

