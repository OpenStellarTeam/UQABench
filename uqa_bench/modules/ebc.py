#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from typing import Dict, Tuple

import torch
import torch.nn as nn

from uqa_bench.config import DataConfig


def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x


class EmbeddingBagCollection(nn.Module):
    def __init__(self, conf: DataConfig):
        super().__init__()
        self.conf = conf
        self.bags = nn.ModuleDict({
            k: nn.EmbeddingBag(
                v.vocab_size,
                v.embed_dim,
                mode="mean",
                include_last_offset=True
            )
            for k, v in conf.tables_dict.items()
        })
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for _, v in self.bags.items():
            truncated_normal(v.weight, 0, 0.02)

    def forward(self, ids_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        output = {}
        for name in self.conf.feature_names:
            tab_name = self.conf.feats_dict[name].table_name
            ids, offsets = ids_dict[name]
            emb = self.bags[tab_name](ids, offsets=offsets)
            output[name] = emb
        return output
