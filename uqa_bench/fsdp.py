#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

SHARD_STRATEGY_MAPPINGS = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
}


def get_mixed_prec_conf(fp16: bool, bf16: bool):
    if fp16:
        param_dtype = torch.float16
        reduce_dtype = torch.float16
    elif bf16:
        param_dtype = torch.bfloat16
        reduce_dtype = torch.float32
    else:
        param_dtype = torch.float32
        reduce_dtype = torch.float32
    buffer_dtype = torch.float32

    mixed_prec = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype
    )
    return mixed_prec


def get_wrap_policy(layer_cls):
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=layer_cls
    )
    return auto_wrap_policy


def init_fsdp_model(model: nn.Module, shard_strategy: str, wrap_cls, fp16: bool, bf16: bool):
    auto_wrap_policy = get_wrap_policy(wrap_cls)
    mixed_prec = get_mixed_prec_conf(fp16, bf16)
    shard_strategy = SHARD_STRATEGY_MAPPINGS[shard_strategy]

    model = FSDP(
        model,
        sharding_strategy=shard_strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_prec,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    return model
