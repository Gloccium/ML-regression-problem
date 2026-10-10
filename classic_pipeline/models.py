from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def get_model(model_name, config):
    seed = config.general.seed

    if model_name == "ridge":
        params = config.models.ridge
        return Ridge(alpha=params.alpha)

    if model_name == "random_forest":
        params = config.models.random_forest

        return RandomForestRegressor(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_leaf=params.min_samples_leaf,
            random_state=seed,
            n_jobs=-1
        )

    if model_name == "xgboost":
        from xgboost import XGBRegressor

        params = config.models.xgboost

        return XGBRegressor(
            n_estimators=params.n_estimators,
            learning_rate=params.learning_rate,
            max_depth=params.max_depth,
            subsample=params.subsample,
            colsample_bytree=params.colsample_bytree,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1
        )

    raise ValueError(f"Unknown model: {model_name}")