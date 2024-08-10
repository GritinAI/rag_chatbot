# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np


__all__ = ["RAG"]


class RAG(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.vector_database = None

    def forward(self, x, k=5):
        return torch.concat([x, self.mips(x, k=k)], dim=-1)

    def mips(self, query, k=5):
        return self.vector_database(query, k=k)
