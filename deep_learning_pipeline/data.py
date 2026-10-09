import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

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

class HousePricesDataset(Dataset):
    def __init__(self, X, y=None):
        X = np.ascontiguousarray(
            X,
            dtype=np.float32,
        )

        self.X = torch.from_numpy(X)

        if y is not None:
            y = np.ascontiguousarray(
                np.asarray(y),
                dtype=np.float32,
            )

            self.y = torch.from_numpy(y)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is None:
            return self.X[index]

        return self.X[index],self.y[index]


def drop_outliers(df):
    outlier_mask = (
        (df["GrLivArea"] > 4000) &
        (df["SalePrice"] < 300000)
    )

    return df.loc[~outlier_mask].copy()


def fill_semantic_missing_columns(df):
    df = df.copy()

    for col in SEMANTIC_MISSING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    return df


def create_preprocessor(X_train):
    num_columns = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_columns = X_train.select_dtypes(exclude=np.number).columns.tolist()

    num_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    cat_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            num_pipeline,
            num_columns,
        ),
        (
            "categorical",
            cat_pipeline,
            cat_columns,
        ),
    ])

    return preprocessor


def load_data(config):
    train_df = pd.read_csv(config.paths.train_data)
    test_df = pd.read_csv(config.paths.test_data)

    if config.data.drop_outliers:
        train_df = drop_outliers(train_df)

    train_df = fill_semantic_missing_columns(train_df)
    test_df = fill_semantic_missing_columns(test_df)

    target = config.data.target
    id_column = config.data.id_column

    X = train_df.drop(columns=[target, id_column])
    y = train_df[target].copy()

    test_ids = test_df[id_column].copy()
    X_test = test_df.drop(columns=[id_column])

    if config.data.target_transform == "log1p":
        y = np.log1p(y)

    return X, y, X_test, test_ids


def create_fold_dataloaders(X, y, X_test, train_idx, val_idx, config):
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()

    if config.data.standardize_target:
        target_mean = float(y_train.mean())
        target_std = float(y_train.std(ddof=0))

        y_train = (y_train - target_mean) / target_std
        y_val = (y_val - target_mean) / target_std
    else:
        target_mean = 0.0
        target_std = 1.0

    preprocessor = create_preprocessor(X_train)

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test_fold = preprocessor.transform(X_test)

    train_dataset = HousePricesDataset(X_train, y_train)
    val_dataset = HousePricesDataset(X_val, y_val)
    test_dataset = HousePricesDataset(X_test_fold)

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = config.training.batch_size, shuffle=False)

    input_size = X_train.shape[1]

    return train_loader, val_loader, test_loader, preprocessor, input_size, target_mean, target_std