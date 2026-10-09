from deep_learning_pipeline.config import config
from deep_learning_pipeline.data import create_dataloaders
from deep_learning_pipeline.model import MLPRegressor
from deep_learning_pipeline.train import train
from deep_learning_pipeline.utils import (
    set_seed,
    get_device,
    get_loss,
    get_metric,
    get_optimizer,
    get_scheduler,
    log_experiment
)


def fit(config):
    set_seed(config.general.seed)
    device = get_device(config)

    train_loader, val_loader, test_loader, preprocessor, test_ids, input_size = create_dataloaders(config)

    model = MLPRegressor(
        input_size=input_size,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
    ).to(device)

    loss = get_loss(config)
    metric = get_metric(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    log_experiment(
        config=config,
        device=device,
        input_size=input_size,
        loss_func=loss,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Тренировка модели
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_func=loss,
        metric_func=metric,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
    )

    return (
        model,
        history,
        test_loader,
        test_ids,
        preprocessor,
    )

if __name__ == "__main__":
    model, history, test_loader, test_ids, preprocessor = fit(config)