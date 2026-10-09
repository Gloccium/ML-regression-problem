import copy
import torch


def validate(model, val_loader, loss_func, metric_func, device, target_mean=0.0, target_std=1.0):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_targets = []

    with torch.inference_mode():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            predictions = model(X)
            loss = loss_func(predictions, y)

            batch_size = X.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_predictions.append(predictions.cpu())
            all_targets.append(y.cpu())

        val_loss = total_loss / total_samples

        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        metric_predictions = all_predictions * target_std + target_mean
        metric_targets = all_targets * target_std + target_mean

        val_metric = metric_func(metric_predictions, metric_targets).item()

    return val_loss, val_metric


def train(model, train_loader, val_loader, loss_func, metric_func, optimizer, scheduler, device, config, target_mean=0.0, target_std=1.0):
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_metric": [],
        "learning_rate": [],
    }

    for epoch in range(1, config.training.epochs +1):
        model.train()

        total_train_loss = 0.0
        total_samples = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(X)

            loss = loss_func(predictions, y)
            loss.backward()

            optimizer.step()
            batch_size = X.size(0)

            total_train_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_train_loss / total_samples
        val_loss, val_metric = validate(
            model=model,
            val_loader=val_loader,
            loss_func=loss_func,
            metric_func=metric_func,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch}/{config.training.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val {config.training.metric.upper()}: {val_metric:.4f} | "
            f"Learning rate: {current_lr:.6f}"
        )

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            best_model_state = copy.deepcopy(model.state_dict())

        else:
            epochs_without_improvement +=1

        # Ранняя остановка, когда val loss перестает улучшаться
        if epochs_without_improvement >= config.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} | "
                  f"Best val loss is: {best_val_loss:.4f}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history