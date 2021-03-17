from typing import Any
from torch import nn


class Straight(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, *args: Any, **kwargs: Any):
        return input
