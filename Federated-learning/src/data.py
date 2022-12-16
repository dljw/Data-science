import pandas as pd
import numpy as np
import os

import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

CATEGORICAL_FEATURE_KEYS = [
    "Dept",
    "IsHoliday",
    "Super_Bowl",
    "Type",
    "Labor_Day",
    "Thanksgiving",
    "Christmas",
]

NUMERIC_FEATURE_KEYS = [
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "Size",
    "CPI",
]

COLUMNS = [
    "Temperature",
    "Size",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Dept_1",
    "Dept_2",
    "Dept_3",
    "Dept_4",
    "Dept_5",
    "Dept_6",
    "Dept_7",
    "Dept_8",
    "Dept_9",
    "Dept_10",
    "Dept_11",
    "Dept_12",
    "Dept_13",
    "Dept_14",
    "Dept_16",
    "Dept_17",
    "Dept_18",
    "Dept_19",
    "Dept_20",
    "Dept_21",
    "Dept_22",
    "Dept_23",
    "Dept_24",
    "Dept_25",
    "Dept_26",
    "Dept_27",
    "Dept_28",
    "Dept_29",
    "Dept_30",
    "Dept_31",
    "Dept_32",
    "Dept_33",
    "Dept_34",
    "Dept_35",
    "Dept_36",
    "Dept_37",
    "Dept_38",
    "Dept_39",
    "Dept_40",
    "Dept_41",
    "Dept_42",
    "Dept_44",
    "Dept_45",
    "Dept_46",
    "Dept_47",
    "Dept_48",
    "Dept_49",
    "Dept_50",
    "Dept_51",
    "Dept_52",
    "Dept_54",
    "Dept_55",
    "Dept_56",
    "Dept_58",
    "Dept_59",
    "Dept_60",
    "Dept_65",
    "Dept_67",
    "Dept_71",
    "Dept_72",
    "Dept_74",
    "Dept_77",
    "Dept_78",
    "Dept_79",
    "Dept_80",
    "Dept_81",
    "Dept_82",
    "Dept_83",
    "Dept_85",
    "Dept_87",
    "Dept_90",
    "Dept_91",
    "Dept_92",
    "Dept_93",
    "Dept_94",
    "Dept_95",
    "Dept_96",
    "Dept_97",
    "Dept_98",
    "Dept_99",
    "IsHoliday_False",
    "IsHoliday_True",
    "Super_Bowl_True",
    "Super_Bowl_False",
    "Type_A",
    "Type_B",
    "Type_C",
    "Labor_Day_False",
    "Labor_Day_True",
    "Thanksgiving_True",
    "Thanksgiving_False",
    "Christmas_True",
    "Christmas_False",
]


def load_data(client="1", data_dir="../data/store_level_data"):
    path = os.path.join(data_dir, "store_" + client)
    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    x_train = train.drop(["Weekly_Sales"], axis=1)
    y_train = train["Weekly_Sales"]

    x_test = test.drop(["Weekly_Sales"], axis=1)
    y_test = test["Weekly_Sales"]

    ct = ColumnTransformer(
        [
            ("Numeric", StandardScaler(), NUMERIC_FEATURE_KEYS),
            (
                "Categorical",
                # OneHotEncoder(handle_unknown="infrequent_if_exist"),
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURE_KEYS,
            ),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", ct)])
    X_train_trans = pipeline.fit_transform(x_train)
    X_test_trans = pipeline.transform(x_test)
    trans_col = (
        pipeline.named_steps["preprocessor"]
        .transformers_[1][1]
        .get_feature_names_out(CATEGORICAL_FEATURE_KEYS)
    )
    X_train_nn = X_train_trans.toarray()
    X_train_nn = pd.DataFrame(
        X_train_nn, columns=NUMERIC_FEATURE_KEYS + trans_col.tolist()
    )

    X_test_nn = X_test_trans.toarray()
    X_test_nn = pd.DataFrame(
        X_test_nn, columns=NUMERIC_FEATURE_KEYS + trans_col.tolist()
    )

    for column in COLUMNS:
        if column not in X_train_nn.columns:
            X_train_nn[column] = 0
        if column not in X_test_nn.columns:
            X_test_nn[column] = 0

    X_train_nn = tf.convert_to_tensor(X_train_nn)
    y_train_nn = tf.convert_to_tensor(y_train)
    X_test_nn = tf.convert_to_tensor(X_test_nn)
    y_test_nn = tf.convert_to_tensor(y_test)

    return X_train_nn, y_train_nn, X_test_nn, y_test_nn
