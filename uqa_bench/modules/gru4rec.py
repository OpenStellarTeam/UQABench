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
import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
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
        embedding_dim: int,
        num_layers: int,
        hidden_size: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.gru_layers = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            nn.init.xavier_uniform_(module.weight_hh_l0)
            nn.init.xavier_uniform_(module.weight_ih_l0)

    def forward(
        self,
        user_embeddings: torch.Tensor,
        past_lengths: torch.Tensor,
    ):
        output, _ = self.gru_layers(user_embeddings)
        output = self.dropout(output)
        output = self.dense(output)
        output = self.norm(output)
        return output
