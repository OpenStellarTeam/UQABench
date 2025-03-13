#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from typing import Dict, Tuple, Optional, Union, List

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from uqa_bench.config import DataConfig
from uqa_bench.modules.hstu import HSTU
from uqa_bench.modules.sasrec import SASRec
from uqa_bench.modules.mamba import MambaConfig, MambaModel
from uqa_bench.modules.gru4rec import GRU4Rec
from uqa_bench.modules.transformer import TransformerPlus, TransformerConfig
from uqa_bench.modules.ebc import EmbeddingBagCollection

from uqa_bench.models.qwen2 import (
    Qwen2Config,
    Qwen2Model,
    Qwen2ForCausalLM,
)


class SimplePooler(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(in_dim, out_dim)
        self.output_length = 1

    def forward(self, x, seq_lengths):
        x = self.up_proj(x)
        device = x.device
        batch_size, seq_len, _ = x.shape

        mask = torch.arange(
            seq_len, device=device
        ).expand(batch_size, seq_len) < seq_lengths.unsqueeze(1)

        masked = x * mask.unsqueeze(-1).type_as(x)

        sumx = masked.sum(dim=1)

        pooled = sumx / seq_lengths.unsqueeze(1)

        return pooled.unsqueeze(1)


class Qwen2ForUQA(Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config, data_conf: DataConfig, max_seq_len: int):
        super().__init__(config)
        self.names = data_conf.feature_names
        self.ue_hidden_size = 512
        self.model = Qwen2Model(config)
        self.max_seq_len = max_seq_len

        embed_dim = data_conf.embedding_dim
        self.ebc = EmbeddingBagCollection(data_conf)
        self.proj1 = nn.Linear(embed_dim, self.ue_hidden_size)
        self.vocab_size, self.item_dim = self.ebc.bags["item_id"].weight.shape
        self.backbone = self._build_backbone()
        self.adapter = SimplePooler(
            self.ue_hidden_size, self.config.hidden_size
        )
        self.loss_offset = self.adapter.output_length

    def _build_backbone(self):
        raise NotImplementedError()

    def prefill_memory(
        self,
        features: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        seq_lengths: torch.Tensor,
    ):
        batch_size = seq_lengths.shape[0]
        embs = []
        emb_dict = self.ebc(features)
        for name in self.names:
            emb = emb_dict[name].view(batch_size, self.max_seq_len, -1)
            embs.append(emb)
        embeddings = self.proj1(torch.cat(embs, dim=2))
        hiddens = self.backbone(embeddings, seq_lengths)
        user_embeds = self.adapter(hiddens, seq_lengths)
        return user_embeds

    def forward(
        self,
        features: Optional[Dict[str,
                                Tuple[torch.Tensor, torch.Tensor]]] = None,
        seq_lengths: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        user_embeds: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if user_embeds is None:
            batch_size = seq_lengths.shape[0]
            embs = []
            emb_dict = self.ebc(features)
            for name in self.names:
                emb = emb_dict[name].view(batch_size, self.max_seq_len, -1)
                embs.append(emb)
            embeddings = self.proj1(torch.cat(embs, dim=2))
            hiddens = self.backbone(embeddings, seq_lengths)
            user_embeds = self.adapter(hiddens, seq_lengths)
        word_embeds = self.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([user_embeds, word_embeds], dim=1)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., self.loss_offset:-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HSTUForUQA(Qwen2ForUQA):
    def _build_backbone(self, *args, **kwargs):
        return HSTU(
            self.max_seq_len, self.ue_hidden_size, 4, 8, 256, 256, "rel_bias", "uvqk", "silu", 0.1, 0.0, verbose=False
        )


class SASRecForUQA(Qwen2ForUQA):
    def _build_backbone(self, *args, **kwargs):
        return SASRec(
            self.max_seq_len, self.ue_hidden_size, 6, 8, 1536, 0.1
        )


class TrmPlusForUQA(Qwen2ForUQA):
    def _build_backbone(self, *args, **kwargs):
        config = TransformerConfig(hidden_size=512)
        return TransformerPlus(config)


class GRU4RecForUQA(Qwen2ForUQA):
    def _build_backbone(self, *args, **kwargs):
        model = GRU4Rec(
            embedding_dim=self.ue_hidden_size,
            num_layers=2,
            hidden_size=int(self.ue_hidden_size * 2),
            dropout_rate=0.1,
        )
        return model


class Mamba4RecForUQA(Qwen2ForUQA):
    def _build_backbone(self, *args, **kwargs):
        config = MambaConfig(
            d_model=self.ue_hidden_size,
            n_layer=4,
            ssm_cfg={"layer": "Mamba2"}
        )
        model = MambaModel(config)
        return model
