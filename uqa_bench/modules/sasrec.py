# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Implements SASRec (Self-Attentive Sequential Recommendation, https://arxiv.org/abs/1808.09781, ICDM'18).

Compared with the original paper which used BCE loss, this implementation is modified so that
we can utilize a Sampled Softmax loss proposed in Revisiting Neural Retrieval on Accelerators
(https://arxiv.org/abs/2306.04039, KDD'23) and Turning Dross Into Gold Loss: is BERT4Rec really
better than SASRec? (https://arxiv.org/abs/2309.07602, RecSys'23), where the authors showed
sampled softmax loss to significantly improved SASRec model quality.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRec(nn.Module):
    """
    Implements SASRec (Self-Attentive Sequential Recommendation, https://arxiv.org/abs/1808.09781, ICDM'18).

    Compared with the original paper which used BCE loss, this implementation is modified so that
    we can utilize a Sampled Softmax loss proposed in Revisiting Neural Retrieval on Accelerators
    (https://arxiv.org/abs/2306.04039, KDD'23) and Turning Dross Into Gold Loss: is BERT4Rec really
    better than SASRec? (https://arxiv.org/abs/2309.07602, RecSys'23), where the authors showed
    sampled softmax loss to significantly improved SASRec model quality.
    """

    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            embedding_dim, num_heads, 
            dim_feedforward=ffn_hidden_dim, dropout=dropout_rate, batch_first=True
        )
        self.position_embedding = nn.Embedding(max_sequence_len, embedding_dim)
        self.transformer = nn.TransformerEncoder(layer, num_blocks)
        self.norm = nn.LayerNorm(embedding_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        user_embeddings: torch.Tensor,
        past_lengths: torch.Tensor,
    ):
        bsz, seq_len = user_embeddings.shape[:2]
        device = user_embeddings.device
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)
        position_embedding = self.position_embedding(position_ids)

        user_embeddings = user_embeddings + position_embedding

        mask = torch.arange(
            seq_len, device=device
        ).expand(bsz, seq_len) < past_lengths.unsqueeze(1)
        mask = ~mask
        src_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, user_embeddings.device, user_embeddings.dtype
        )
        out = self.transformer(
            user_embeddings, 
            mask=src_mask, 
            src_key_padding_mask=mask, 
            is_causal=True
        )
        out = self.norm(out)
        return out
