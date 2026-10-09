import numpy as np
import torch


def predict(model, data_loader, device, target_mean=0.0, target_std=1.0):
    model.eval()

    all_predictions = []

    with torch.inference_mode():
        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                X = batch[0]
            else:
                X = batch

            X = X.to(device)

            predictions = model(X)
            predictions = predictions * target_std + target_mean

            all_predictions.append(predictions.cpu().numpy())

    return np.concatenate(all_predictions)