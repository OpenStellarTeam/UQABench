#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2025/01/10 14:21:23
import os
import json
import random
import argparse
from pathlib import Path

import torch
import torch.distributed as dist

from torch.utils.data import DataLoader, SequentialSampler

from uqa_bench.config import DATA_CONFIGS
from uqa_bench.dataset import get_align_dataset
from uqa_bench.dataloader import SeqModelAlignEvalDataCollator
from uqa_bench.models.align import (
    HSTUForUQA,
    SASRecForUQA,
    TrmPlusForUQA,
    GRU4RecForUQA,
    Mamba4RecForUQA,
)
from uqa_bench.models.qwen2 import Qwen2TokenizerFast


def setup():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
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
        "--model_path", type=str, default="xx"
    )
    parser.add_argument(
        "--output_path", type=str, required=True
    )
    parser.add_argument("--max_text_len", type=int, default=256)
    args = parser.parse_args()
    return args


def main():
    setup()
    args = get_args()
    conf = DATA_CONFIGS["uqa_bench"]
    max_len = conf.max_len
    max_text_len = args.max_text_len
    _, eval_dataset = get_align_dataset(conf)
    tokenizer = Qwen2TokenizerFast.from_pretrained(args.model_path)
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
    device = torch.device("cuda")
    model = CLAS.from_pretrained(
        args.model_path, conf, max_len,
        device_map=device
    )
    model.eval()

    sampler = SequentialSampler(eval_dataset)
    collate_fn = SeqModelAlignEvalDataCollator(
        tokenizer, conf, max_len, max_text_len
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        sampler=sampler,
        pin_memory=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    eos_id = model.config.eos_token_id
    pad_id = model.config.eos_token_id
    output_path = Path(args.output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    sink = open(args.output_path, "w")
    step = 0
    num_steps = len(eval_dataloader)
    for batch in eval_dataloader:
        batch_features, batch_lengths, input_ids, questions, answers = batch
        batch_lengths = batch_lengths.to(device=device, non_blocking=True)
        input_ids = input_ids.to(device=device, non_blocking=True)
        for k in batch_features:
            values, offsets = batch_features[k]
            batch_features[k] = (
                values.to(device=device, non_blocking=True),
                offsets.to(device=device, non_blocking=True)
            )
        generated_ids = []
        user_memory = model.prefill_memory(
            batch_features, batch_lengths
        )
        for i in range(1):
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    user_embeds=user_memory
                )
            logits = outputs.logits[:, -1, :]
            new_input_ids = logits.argmax(dim=1)
            if new_input_ids[0] == eos_id or new_input_ids[0] == pad_id or new_input_ids[0] == 151645:
                break
            input_ids = torch.cat(
                [input_ids, new_input_ids.unsqueeze(1)], dim=1)
            generated_ids.append(new_input_ids.item())
        generated_text = tokenizer.decode(
            generated_ids, skip_special_tokens=False)
        print(f"------- step {step}/{num_steps} --------")
        print(" question:", questions[0])
        print("   answer:", answers[0])
        print("generated:", generated_text)
        msg = {
            "question": questions[0],
            "prediction": generated_text,
            "answer": answers[0],
        }
        sink.write(json.dumps(msg) + "\n")
        step += 1
        if step % 10 == 0:
            print(f"Step {step}/{num_steps}")
            sink.flush()
    sink.close()


if __name__ == "__main__":
    main()
