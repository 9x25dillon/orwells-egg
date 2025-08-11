from typing import List
import torch
import torch.nn as nn
from ml2_core import GradNormalizer


def apply_train_step(model_layers: List[nn.Module], loss: torch.Tensor, optimizer: torch.optim.Optimizer, norm: str = "max") -> None:
    loss.backward()
    normalizer = GradNormalizer(model_layers, norm=norm)
    normalizer.apply()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)