#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from typing import List

import pyarrow as pa
from torch.utils.data import Dataset

from uqa_bench.config import DataConfig


class ArrowDataset(Dataset):
    def __init__(self, filename: str, cols: List[str]) -> None:
        super().__init__()
        self.filename = filename
        self.cols = cols
        with pa.memory_map(filename, "r") as source:
            self.reader = pa.ipc.open_file(source).read_all()

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int):
        ret = {}
        for col in self.cols:
            ret[col] = self.reader[col][index].as_py()
        return ret


def get_pretrain_dataset(conf: DataConfig):
    train_path = "datas/pretrain_data_train.arrow"
    eval_path = "datas/pretrain_data_eval.arrow"
    train_dset = ArrowDataset(train_path, conf.feature_names)
    eval_dset = ArrowDataset(eval_path, conf.feature_names)
    return train_dset, eval_dset


def get_align_dataset(conf: DataConfig):
    train_path = "datas/align_data_train.arrow"
    eval_path = "datas/align_data_eval.arrow"
    train_dset = ArrowDataset(train_path, conf.feature_names + ["text"])
    eval_dset = ArrowDataset(eval_path, conf.feature_names + ["text"])
    return train_dset, eval_dset