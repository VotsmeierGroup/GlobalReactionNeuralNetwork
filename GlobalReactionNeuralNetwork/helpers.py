import torch
from typing import Dict
from GlobalReactionNeuralNetwork.load_data import Dataset

# RMS relative error used for model training
class RMS_RelativeError(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        relative_error = torch.abs((predictions - targets) / targets) * 100
        return torch.sqrt(torch.mean(relative_error**2))

# Relative error used for model evaluation
class RelativeError(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        relative_error = torch.abs((predictions - targets) / targets) * 100
        return torch.mean(relative_error)

# Helper to check model accuracy during training
def check_accuracy(
    model: torch.nn.Module,
    dataset: Dataset,
    criterion: torch.nn.Module,
    error_metric: torch.nn.Module
) -> Dict[str, float]:
    with torch.no_grad():
        ypred = model(dataset.x).squeeze()
        error_val = error_metric(ypred, dataset.y).item()
        criterion_val = criterion(ypred, dataset.y).item()

    return {
        'error_metric': error_val,
        'criterion': criterion_val
    }

