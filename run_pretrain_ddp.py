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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from uqa_bench.config import DATA_CONFIGS
from uqa_bench.dataset import get_pretrain_dataset
from uqa_bench.dataloader import SeqModelPreTrainDataCollator
from uqa_bench.models.pretrain import (
    HSTUForPretrain,
    SASRecForPretrain,
    TrmPlusForPretrain,
    GRU4RecForPretrain,
    Mamba4RecForPretrain
)
from uqa_bench.metrics import recalls_and_ndcgs_for_ks


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.getenv("LOCAL_RANK"))
    random_seed = 42
    random.seed(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)


def print_rank0(*args, **kwargs):
    rank = int(os.getenv("RANK"))
    if rank == 0:
        print(*args, **kwargs)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=["hstu", "sasrec", "trm_plus", "gru4rec", "mamba"], default="hstu"
    )
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    args = parser.parse_args()
    return args


def get_num_parameters(model: torch.nn.Module) -> Dict[str, str]:
    n_params = sum(p.numel() for p in model.parameters())
    n_tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tr_dense_params = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("backbone.")
    )
    info = {
        "total": f"{n_params:,}",
        "trainable": f"{n_tr_params:,}",
        "trainable_backbone": f"{n_tr_dense_params:,}",
    }
    return info


def evaluate(model, device, dataloader: DataLoader, epoch: int):
    torch.cuda.empty_cache()
    model.eval()
    all_preds = []
    all_labels = []
    num_eval_steps = len(dataloader)
    step = 0
    for batch in dataloader:
        batch_features, batch_lengths, batch_labels = batch
        batch_lengths = batch_lengths.to(device=device, non_blocking=True)
        batch_labels = batch_labels.to(device=device, non_blocking=True)
        for k in batch_features:
            values, offsets = batch_features[k]
            batch_features[k] = (
                values.to(device=device, non_blocking=True),
                offsets.to(device=device, non_blocking=True)
            )
        with torch.no_grad():
            logits = model(batch_features, batch_lengths)
        logits = logits.softmax(dim=-1)

        inds = torch.arange(batch_lengths.shape[0])
        # Only Eval last token
        valid_preds = logits[inds, batch_lengths - 2]
        valid_labels = batch_labels[inds, batch_lengths - 2]

        # Eval all tokens
        # valid_preds = logits[batch_labels != -100]
        # valid_labels = batch_labels[batch_labels != -100]
        all_preds.extend(valid_preds.cpu())
        all_labels.extend(valid_labels.cpu())

        if step % 100 == 0:
            print_rank0(f"Eval step {step} / {num_eval_steps}")
        step += 1

    all_preds = torch.stack(all_preds)
    all_labels = torch.stack(all_labels)
    all_labels = F.one_hot(all_labels, num_classes=all_preds.shape[1])
    results = {}
    metrics = recalls_and_ndcgs_for_ks(
        all_preds, all_labels, [1, 5, 10, 20, 50, 100]
    )
    for k in (
        "Recall@1", "Recall@5", "Recall@10", "Recall@20", "Recall@50", "Recall@100",
        "NDCG@1", "NDCG@5", "NDCG@10", "NDCG@20", "NDCG@50", "NDCG@100",
    ):
        metric = torch.tensor(metrics[k], device=device)
        dist.all_reduce(metric, dist.ReduceOp.AVG)
        results[k] = metric.cpu().item()
    print_rank0(f"Eval metrics at epoch {epoch}:", results)
    torch.cuda.empty_cache()


def main():
    setup()
    args = get_args()
    print_rank0(args)
    conf = DATA_CONFIGS["uqa_bench"]
    max_len = conf.max_len
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    train_dataset, eval_dataset = get_pretrain_dataset(conf)

    device = torch.device("cuda")
    rank = int(os.getenv("RANK"))

    train_sampler = DistributedSampler(train_dataset)
    collate_fn = SeqModelPreTrainDataCollator(conf, max_len)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=0 if args.model == "mamba" else 8,
        collate_fn=collate_fn
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=4,
        sampler=DistributedSampler(eval_dataset, shuffle=False),
        pin_memory=True,
        num_workers=0 if args.model == "mamba" else 8,
        collate_fn=collate_fn
    )
    if args.model == "hstu":
        model = HSTUForPretrain(conf, max_len)
    elif args.model == "sasrec":
        model = SASRecForPretrain(conf, max_len)
    elif args.model == "trm_plus":
        model = TrmPlusForPretrain(conf, max_len)
    elif args.model == "gru4rec":
        model = GRU4RecForPretrain(conf, max_len)
    elif args.model == "mamba":
        model = Mamba4RecForPretrain(conf, max_len)

    print_rank0("Num params: ", get_num_parameters(model))
    model.to(device)
    model = DDP(model, find_unused_parameters=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    step = 0
    num_steps = num_epochs * len(train_dataloader)
    print_rank0(model)
    loss_buf = torch.zeros(1, device=device)
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch_features, batch_lengths, batch_labels = batch
            batch_lengths = batch_lengths.to(device=device, non_blocking=True)
            batch_labels = batch_labels.to(device=device, non_blocking=True)
            for k in batch_features:
                values, offsets = batch_features[k]
                batch_features[k] = (
                    values.to(device=device, non_blocking=True),
                    offsets.to(device=device, non_blocking=True)
                )
            loss = model(batch_features, batch_lengths, batch_labels)
            loss.backward()
            optimizer.step()
            loss_buf.add_(loss.detach())
            step += 1
            if step % 100 == 0:
                msg = "[{} {}/{}] ".format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    step,
                    num_steps
                )
                msg += f"loss: {(loss_buf / 100).item()}"
                loss_buf.zero_()
                print_rank0(msg)
        evaluate(model, device, eval_dataloader, epoch)
        if rank == 0:
            states = model.module.state_dict()
            save_path = output_dir / f"epoch_{epoch}.pt"
            torch.save(states, save_path)


if __name__ == "__main__":
    main()
