from typing import Dict
import torch
from torch import Tensor
from torchmetrics.functional import accuracy, auroc, precision, recall, f1_score


def transform_preds(t: Tensor) -> Tensor:
    # transform preds tensor into multiclass form to calculate f1-macro
    t = torch.cat([1 - t, t], dim=1)

    return t


def calculate_metrics(preds: Tensor, target: Tensor) -> Dict:
    metrics = {
        'acc': accuracy(preds=preds, target=target),
        'precision': precision(preds, target),
        'recall': recall(preds, target),
        'auc': auroc(preds, target),
        'f1_micro': f1_score(preds, target, average='micro', threshold=0.5)
    }

    preds = transform_preds(preds)

    metrics.update({
        'f1_macro': f1_score(preds, target, average='macro', num_classes=2)
    })

    return metrics
