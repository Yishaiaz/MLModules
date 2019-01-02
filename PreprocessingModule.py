import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def importing_the_dataset(name: str = "data.csv") -> tuple:
    try:
        dataset = pd.read_csv(name)
        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1:].values
        return x, y
    except Exception as e:
        print(e)
        return None


'''
x columns must begin with the ones that are missing data, all the columns with intact complete data are last.
'''


def fill_up_missing_data(dataset_x=[[]],
                         dataset_x_start_of_missing_data_index=0,
                         dataset_x_end_of_missing_data_index=1,
                         strategy="mean",
                         missing_value_type: str = "NaN"):
    imputer = Imputer(missing_values=missing_value_type,
                      strategy=strategy, axis=0)
    imputer = imputer.fit(X=dataset_x[:, dataset_x_start_of_missing_data_index:dataset_x_end_of_missing_data_index])
    dataset_x[:, dataset_x_start_of_missing_data_index:dataset_x_end_of_missing_data_index] \
        = imputer.transform(dataset_x[:, dataset_x_start_of_missing_data_index:dataset_x_end_of_missing_data_index])
    return dataset_x




'''
this function takes one column at a time and encodes it to dummy encoding:
i.e this column [france,
                 italy,
                 spain] will be transformed to:
[ [1,0,0],
  [0,1,0],
  [0,0,1] ]

if dummy encoding is not needed , meaning there is a value difference to 
different categories (i.e - shirt sizes, medium IS smaller than Large)
pass in the strategy parameter as "encoding"
pass in the entire X dataset, excluding the Y column.
if you would like to encode your result column (y), pass on is_results = True.
Warning : if you are encoding the results column, pass param strategy ="encoding" 
'''


def encoding_categorical_variables_by_column(dataset: list = [],
                                             categorical_columns: list = [0],
                                             strategy: str = "dummy_encoding",
                                             is_results: bool = False):
    label_encoder = LabelEncoder()
    if is_results:
        dataset[:, 0] = label_encoder.fit_transform(dataset)
    else:
        for index in categorical_columns:
            dataset[:, index] = label_encoder.fit_transform(dataset[:, index])
    if strategy is "dummy_encoding":
        for index in categorical_columns:
            templist = [index]
            one_hot = OneHotEncoder(categorical_features=templist)
            dataset = one_hot.fit_transform(dataset).toarray()
        return dataset
    return dataset


def split_to_train_and_test(x_dataset,
                            y_dataset,
                            random_state=42,
                            test_size: float = 0.2) -> tuple:
    x_train, x_test, y_train, y_test = train_test_split(x_dataset,
                                                        y_dataset,
                                                        test_size=test_size,
                                                        random_state=random_state)
    return x_train, x_test, y_train, y_test




