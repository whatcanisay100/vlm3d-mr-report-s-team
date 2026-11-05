from __future__ import annotations

from itertools import repeat

import torch


def to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return tuple(repeat(value, 2))


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def drop_path(value, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return value
    keep_prob = 1.0 - drop_prob
    shape = (value.shape[0],) + (1,) * (value.ndim - 1)
    random_tensor = value.new_empty(shape).bernoulli_(keep_prob)
    return value.div(keep_prob) * random_tensor


class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = float(drop_prob or 0.0)

    def forward(self, value):
        return drop_path(value, self.drop_prob, self.training)
