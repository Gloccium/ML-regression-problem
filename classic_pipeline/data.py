import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEMANTIC_MISSING_COLUMNS = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
]


def drop_outliers(df):
    outlier_mask = (
        (df["GrLivArea"] > 4000)
        & (df["SalePrice"] < 300000)
    )

    return df.loc[~outlier_mask].copy()


def fill_semantic_missing_columns(df):
    df = df.copy()

    for column in SEMANTIC_MISSING_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("None")

    return df


def load_train_data(config):
    train_df = pd.read_csv(config.paths.train_data)

    if config.data.drop_outliers:
        train_df = drop_outliers(train_df)

    train_df = fill_semantic_missing_columns(train_df)

    target = config.data.target
    id_column = config.data.id_column

    X = train_df.drop(columns=[target, id_column])

    y = train_df[target].copy()

    if config.data.target_transform == "log1p":
        y = np.log1p(y)

    return X, y


def load_test_data(config):
    test_df = pd.read_csv(config.paths.test_data)

    test_df = fill_semantic_missing_columns(test_df)

    id_column = config.data.id_column

    test_ids = test_df[id_column].copy()

    X_test = test_df.drop(columns=[id_column])

    return X_test, test_ids


def create_preprocessor(X_train):
    numerical_columns = (
        X_train
        .select_dtypes(include=["number"])
        .columns
    )

    categorical_columns = (
        X_train
        .select_dtypes(exclude=["number"])
        .columns
    )

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ])

    return preprocessor