#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from uqa_bench.config import DataConfig
from uqa_bench.modules.hstu import HSTU
from uqa_bench.modules.sasrec import SASRec
from uqa_bench.modules.gru4rec import GRU4Rec
from uqa_bench.modules.mamba import MambaConfig, MambaModel
from uqa_bench.modules.transformer import TransformerPlus, TransformerConfig
from uqa_bench.modules.ebc import EmbeddingBagCollection


class SeqModelForPretrain(nn.Module):
    def __init__(self, conf: DataConfig, max_seq_len: int):
        super().__init__()
        hidden_size = 512
        embed_dim = conf.embedding_dim
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.names = conf.feature_names
        self.ebc = EmbeddingBagCollection(conf)
        self.proj1 = nn.Linear(embed_dim, hidden_size)
        self.vocab_size, self.item_dim = self.ebc.bags["item_id"].weight.shape
        self.backbone = self._build_backbone()
        self.proj2 = nn.Linear(hidden_size, self.item_dim)
        self.score_head_weight = self.ebc.bags["item_id"].weight

    def _build_backbone(self):
        raise NotImplementedError()

    def forward(
        self,
        features: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        seq_lengths: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        batch_size = seq_lengths.shape[0]
        embs = []
        emb_dict = self.ebc(features)
        for name in self.names:
            emb = emb_dict[name].view(batch_size, self.max_seq_len, -1)
            embs.append(emb)
        embeddings = self.proj1(torch.cat(embs, dim=2))
        hiddens = self.backbone(embeddings, seq_lengths)
        hiddens = self.proj2(hiddens)

        logits = F.linear(hiddens, self.score_head_weight)

        if labels is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
        )
        return loss


class HSTUForPretrain(SeqModelForPretrain):
    def _build_backbone(self, *args, **kwargs):
        return HSTU(
            self.max_seq_len, self.hidden_size, 4, 8, 256, 256, "rel_bias", "uvqk", "silu", 0.1, 0.0, verbose=False
        )


class SASRecForPretrain(SeqModelForPretrain):
    def _build_backbone(self, *args, **kwargs):
        return SASRec(
            self.max_seq_len, self.hidden_size, 6, 8, 1536, 0.1
        )


class TrmPlusForPretrain(SeqModelForPretrain):
    def _build_backbone(self, *args, **kwargs):
        config = TransformerConfig(hidden_size=512)
        return TransformerPlus(config)


class GRU4RecForPretrain(SeqModelForPretrain):
    def _build_backbone(self, *args, **kwargs):
        model = GRU4Rec(
            embedding_dim=self.hidden_size,
            num_layers=2,
            hidden_size=int(self.hidden_size * 2),
            dropout_rate=0.1,
        )
        return model


class Mamba4RecForPretrain(SeqModelForPretrain):
    def _build_backbone(self, *args, **kwargs):
        config = MambaConfig(
            d_model=self.hidden_size,
            n_layer=4,
            ssm_cfg={"layer": "Mamba2"}
        )
        model = MambaModel(config)
        return model
