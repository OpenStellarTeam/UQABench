#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/01/10 14:21:23
import os
import random
import datetime
import argparse
from typing import Dict
from pathlib import Path

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

from uqa_bench.config import DATA_CONFIGS
from uqa_bench.dataset import get_align_dataset
from uqa_bench.dataloader import SeqModelAlignDataCollator
from uqa_bench.fsdp import init_fsdp_model, SHARD_STRATEGY_MAPPINGS
from uqa_bench.models.align import (
    HSTUForUQA,
    SASRecForUQA,
    TrmPlusForUQA,
    GRU4RecForUQA,
    Mamba4RecForUQA,
)
from uqa_bench.models.qwen2 import Qwen2TokenizerFast, Qwen2DecoderLayer


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.getenv("LOCAL_RANK"))
    random_seed = 42
    random.seed(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)


def cleanup():
    dist.destroy_process_group()


def print_rank0(*args, **kwargs):
    rank = int(os.getenv("RANK"))
    if rank == 0:
        print(*args, **kwargs)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=["hstu", "sasrec", "trm_plus", "gru4rec", "mamba"], default="hstu"
    )
    parser.add_argument(
        "--llm", type=str, default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument(
        "--ue_ckpt", type=str, default="model.pt"
    )
    parser.add_argument(
        "--fsdp_shard_strategy",
        type=str,
        choices=SHARD_STRATEGY_MAPPINGS.keys(),
        default="shard_grad_op"
    )

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument("--adapter_lr", type=float, default=4e-4)
    parser.add_argument("--other_lr", type=float, default=1e-6)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--only_tune_adapter", action="store_true")
    parser.add_argument("--ue_config_path", type=str, default=None)
    args = parser.parse_args()
    return args


def get_num_parameters(model: torch.nn.Module) -> Dict[str, str]:
    n_params = sum(p.numel() for p in model.parameters())
    n_tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tr_dense_params = sum(p.numel() for n, p in model.named_parameters(
    ) if p.requires_grad and not n.startswith("ebc."))
    info = {
        "total": f"{n_params:,}",
        "trainable": f"{n_tr_params:,}",
        "trainable_dense": f"{n_tr_dense_params:,}",
    }
    return info


def get_optimizer_grouped_params(model, weight_decay, adapter_lr, other_lr):
    decay_params = []
    no_decay_params = []
    adapter_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "adapter" in n:
            adapter_params.append(p)
        elif hasattr(p, "_no_weight_decay") or ".bias" in n:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    print_rank0("num adapter params:", len(adapter_params))
    print_rank0("num wd params:", len(decay_params))
    print_rank0("num no wd params:", len(no_decay_params))
    optimizer_grouped_parameters = [
        {
            "params": adapter_params,
            "weight_decay": weight_decay,
            "lr": adapter_lr
        },
        {
            "params": decay_params,
            "weight_decay": weight_decay,
            "lr": other_lr
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
            "lr": other_lr
        },
    ]
    return optimizer_grouped_parameters


def load_pretrained_user_encoder(model: torch.nn.Module, path):
    def issue_warnings_after_load(load_result):
        if len(load_result.missing_keys) != 0:
            print_rank0(
                f"Missing keys when loading model params: {load_result.missing_keys}."
            )
        if len(load_result.unexpected_keys) != 0:
            print_rank0(
                f"Unexpected keys when loading model params: {load_result.unexpected_keys}."
            )
    states = torch.load(path, map_location="cpu", weights_only=True)
    load_result = model.load_state_dict(states, strict=False)
    # issue_warnings_after_load(load_result)


def main():
    setup()
    args = get_args()
    conf = DATA_CONFIGS["uqa_bench"]
    max_len = conf.max_len
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    max_text_len = args.max_text_len
    train_dataset, _ = get_align_dataset(conf)
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.llm)
    if args.model == "hstu":
        CLAS = HSTUForUQA
    elif args.model == "sasrec":
        CLAS = SASRecForUQA
    elif args.model == "trm_plus":
        CLAS = TrmPlusForUQA
    elif args.model == "gru4rec":
        CLAS = GRU4RecForUQA
    elif args.model == "mamba":
        CLAS = Mamba4RecForUQA
    model: torch.nn.Module = CLAS.from_pretrained(
        args.llm, conf, max_len,
        attn_implementation="flash_attention_2",
    )
    load_pretrained_user_encoder(model, args.ue_ckpt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda")
    rank = int(os.getenv("RANK"))

    train_sampler = DistributedSampler(train_dataset)
    collate_fn = SeqModelAlignDataCollator(
        tokenizer, conf, max_len, max_text_len
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=0 if args.model == "mamba" else 8,
        collate_fn=collate_fn
    )

    if args.only_tune_adapter:
        for n, p in model.named_parameters():
            if "adapter." not in n:
                p.requires_grad = False

    model = init_fsdp_model(
        model, args.fsdp_shard_strategy,
        {Qwen2DecoderLayer}, args.fp16, args.bf16
    )

    grouped_params = get_optimizer_grouped_params(
        model, args.weight_decay, args.adapter_lr,  args.other_lr
    )
    optimizer = optim.AdamW(
        grouped_params,
        lr=args.other_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    scaler = ShardedGradScaler(enabled=args.fp16)

    step = 0
    log_interval = 100
    num_steps = num_epochs * len(train_dataloader)
    print_rank0(model)
    loss_buf = torch.zeros(1, device=device)
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch_features, batch_lengths, input_ids, labels = batch
            batch_lengths = batch_lengths.to(device=device, non_blocking=True)
            input_ids = input_ids.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            for k in batch_features:
                values, offsets = batch_features[k]
                batch_features[k] = (
                    values.to(device=device, non_blocking=True),
                    offsets.to(device=device, non_blocking=True)
                )
            outputs = model(
                features=batch_features,
                seq_lengths=batch_lengths,
                input_ids=input_ids,
                labels=labels
            )
            loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            model.clip_grad_norm_(1.5)
            scaler.step(optimizer)
            scaler.update()
            loss_buf.add_(loss.detach())
            step += 1
            if step % log_interval == 0:
                msg = "[{} {}/{}] ".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    step,
                    num_steps
                )
                msg += f"loss: {(loss_buf / log_interval).item()}"
                loss_buf.zero_()
                print_rank0(msg)
        if epoch == num_epochs - 1 or (epoch + 1) % 3 == 0:
            opt = StateDictOptions(full_state_dict=True, cpu_offload=True)
            states = get_model_state_dict(model, options=opt)
            if rank == 0:
                save_dir = output_dir / f"checkpoint-{step}"
                print_rank0(f"Saving to {str(save_dir)}")
                model.save_pretrained(
                    save_dir, state_dict=states, max_shard_size="3GB"
                )
                tokenizer.save_pretrained(save_dir)
    cleanup()


if __name__ == "__main__":
    main()
