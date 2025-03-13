#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from typing import List
from dataclasses import dataclass


@dataclass
class Feature:
    name: str
    table_name: str


@dataclass
class Table:
    name: str
    vocab_size: int
    embed_dim: int


class DataConfig(object):
    def __init__(self, name: str, feats: List[Feature], tables: List[Table], max_len: int):
        self.feats_dict = {f.name: f for f in feats}
        self.tables_dict = {f.name: f for f in tables}
        self.feature_names = [f.name for f in feats]
        self.embedding_dim = sum(
            self.tables_dict[f.table_name].embed_dim for f in feats
        )
        self.max_len = max_len
        self.name = name


DATA_CONFIGS = {
    "uqa_bench": DataConfig(
        "uqa_bench",
        [
            Feature("uni_seq_item_id", "item_id"),
            Feature("uni_seq_cate_id", "cate_id"),
            Feature("uni_seq_brand_id", "brand_id"),
            Feature("uni_seq_shop_id", "shop_id"),
            Feature("uni_seq_seller_id", "seller_id"),
            Feature("uni_seq_title_bpe_token_ids", "bpe_tokens"),
            Feature("uni_seq_cate_bpe_token_ids", "bpe_tokens"),
            Feature("uni_seq_shop_bpe_token_ids", "bpe_tokens"),
            Feature("uni_seq_brand_bpe_token_ids", "bpe_tokens"),
            Feature("uni_seq_seller_bpe_token_ids", "bpe_tokens")
        ],
        [
            Table("item_id", 994447, 128),
            Table("cate_id", 9330, 64),
            Table("brand_id", 149750, 64),
            Table("shop_id", 268446, 64),
            Table("seller_id", 268446, 64),
            Table("bpe_tokens", 151936, 64),
        ],
        512
    )
}
