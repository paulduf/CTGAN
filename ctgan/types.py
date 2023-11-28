from typing import TypeVar, TypedDict, Tuple
import pandas as pd
import torch

D = TypeVar("D", pd.DataFrame, torch.utils.data.TensorDataset)

T = torch.Tensor  # TypeVar("T", torch.Tensor)


class SolverOptions(TypedDict):
    lr: float
    betas: Tuple[float, float]
    weight_decay: float
