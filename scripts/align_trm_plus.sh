#!/bin/bash

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10737418240
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OMP_NUM_THREADS="16"
export NCCL_DEBUG="WARN"

LAUNCH="torchrun --nproc_per_node 8 --master-port 29503"

ue_ckpt=outputs/trm_plus/epoch_2.pt
output_dir=outputs/trm_plus_align_frozen

$LAUNCH run_align_fsdp.py \
--model trm_plus \
--llm pretrained/qwen25_3b_instruct \
--ue_ckpt=${ue_ckpt} \
--fsdp_shard_strategy full_shard \
--num_epochs 3 \
--batch_size 4 \
--max_text_len 512 \
--output_dir=${output_dir} \
--adapter_lr 1e-4 \
--beta1 0.9 \
--beta2 0.98 \
--weight_decay 0.0 \
--only_tune_adapter \
--bf16