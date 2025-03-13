#!/bin/bash

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10737418240
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OMP_NUM_THREADS="16"
export NCCL_DEBUG="WARN"

LAUNCH="torchrun --nproc_per_node 8 --master-port 29503"

$LAUNCH run_pretrain_ddp.py \
--model trm_plus \
--num_epochs 3 \
--batch_size 4 \
--output_dir=outputs/trm_plus \
--lr 4e-4 \
--beta1 0.9 \
--beta2 0.999 \
--weight_decay 0.0