# UQABench

## Description
The UQABench is a benchmark dataset for evaluating user embeddings in prompting LLMs for personalized question answering. The standardized evaluation process includes **pre-training**, **fine-tuning**, and **evaluating** stages. We provide the requirements and quick-start scripts for each stage.

The source data are user interactions collected and processed from Taobao. Following previous work, we randomly split the data into 9:1 as the training and test sets. The dataset statistics are summarized as follows:

| Data Split    | Total       | #Training  |   #Test    |
|---------------|-------------|------------|------------|
| Interaction   |  31,317,087 | 28,094,799 | 3,222,288  |

Specifically, the training set serves in the pre-training and fine-tuning (aligning) stages. Then, we design task-specific question prompts based on the test set. We refine the questions, filter out low-quality questions, and eventually get 7,192 personalized Q&A for the evaluating stage.


## Download Data & LLM
* Download data from [Kaggle](https://www.kaggle.com/datasets/liulangmingliu/uqabench).
* Download `Qwen/Qwen2.5-3B-Instruct` from [Huggingface](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).

## Requirements
* pytorch 2.4
* fbgemm_gpu
* transformers
* causal_conv1d==1.4.0
* mamba_ssm==2.2.3

## Pretrain
```bash
bash scripts/pretrain_trm_plus.sh
```

## Alignment
```bash
bash scripts/align_trm_plus.sh
```

## Generation
```bash
bash scripts/generate_trm_plus.sh
```

## Evaluation
```bash
python calc_metrics_acc.py generated/trm_plus_align_frozen.jsonl
```
