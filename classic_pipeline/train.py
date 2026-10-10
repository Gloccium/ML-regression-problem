import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from classic_pipeline.data import create_preprocessor
from classic_pipeline.models import get_model


def train_fold(X, y, train_idx, val_idx, model_name, config):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    preprocessor = create_preprocessor(X_train)
    model = get_model(model_name, config)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    val_predictions = pipeline.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

    return pipeline, val_predictions, val_rmse
