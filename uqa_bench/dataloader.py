#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import json
from typing import List, Dict

import torch
import numpy as np

from uqa_bench.config import DataConfig

IGNORE_TOKEN_ID = -100

def row_to_column_2d(row):
    values = []
    offsets1 = [0]
    offsets2 = [0]
    for seq in row:
        offsets2.append(len(seq) + offsets2[-1])
        for feat in seq:
            offsets1.append(len(feat) + offsets1[-1])
            values.extend(feat)
    values = np.array(values, dtype=np.int64)
    offsets1 = np.array(offsets1, dtype=np.int32)
    offsets2 = np.array(offsets2, dtype=np.int32)
    return values, offsets1, offsets2


def preprocess(
    sources,
    tokenizer,
    max_len: int,
    system_message: str = None
) -> Dict:
    """
    2、system and user conversation token not take loss
    """
    roles = {"system":"<|im_start|>system", "user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer('<|im_start|>').input_ids
    im_end = tokenizer('<|im_end|>').input_ids
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        input_id, target = [], []
        if system_message is not None:
            if roles[source[0]["role"]] == roles["system"]:
                source = source[1:]
            else:
                source = source

            system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
            input_id += system
            target += im_start + [IGNORE_TOKEN_ID] * (len(system)-3) + im_end + nl_tokens
            assert len(input_id) == len(target)

        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["content"]).input_ids + im_end + nl_tokens
            input_id += _input_id

            if role == roles["system"]:
                _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end + nl_tokens
            elif role == roles["user"]:
                _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end + nl_tokens
            elif role == roles["assistant"]:
                _target = im_start + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids)+1:-2] + im_end + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids, targets



class SeqModelPreTrainDataCollator(object):
    def __init__(self, conf: DataConfig, max_len: int):
        self.max_len = max_len
        self.names = conf.feature_names

    def __call__(self, examples: List[Dict[str, np.ndarray]]):
        batch_features = {}
        batch_labels = []
        batch_lengths = []
        for feat_name in self.names:
            feats = []
            for e in examples:
                feat = e[feat_name][-self.max_len:]
                feat = feat + [[] for _ in range(self.max_len - len(feat))]
                feats.append(feat)
            values, offests, _ = row_to_column_2d(feats)
            batch_features[feat_name] = (
                torch.from_numpy(values),
                torch.from_numpy(offests),
            )
        for e in examples:
            labels = [
                i[0] for i in e["uni_seq_item_id"][-self.max_len:][1:]
            ] + [-100]
            cur_len = len(labels)
            batch_lengths.append(cur_len)
            labels = labels + [-100] * (self.max_len - cur_len)
            batch_labels.append(labels)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        batch_lengths = torch.tensor(batch_lengths, dtype=torch.long)
        return batch_features, batch_lengths, batch_labels


class SeqModelAlignDataCollator(object):
    def __init__(self, tokenizer, conf: DataConfig, max_len: int, max_text_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_text_len = max_text_len
        self.names = conf.feature_names
        self.system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    
    def __call__(self, examples: List[Dict[str, np.ndarray]]):
        batch_features = {}
        batch_lengths = []
        for feat_name in self.names:
            feats = []
            for e in examples:
                feat = e[feat_name][-self.max_len:]
                if feat_name == "uni_seq_item_id":
                    batch_lengths.append(len(feat))
                feat = feat + [[] for _ in range(self.max_len - len(feat))]
                feats.append(feat)

            values, offests1, _ = row_to_column_2d(feats)
            batch_features[feat_name] = (
                torch.from_numpy(values),
                torch.from_numpy(offests1),
            )
        batch_lengths = torch.tensor(batch_lengths, dtype=torch.long)

        messages = []
        for e in examples:
            message = json.loads(e["text"])
            messages.append(message)
        input_ids, labels = preprocess(messages, self.tokenizer, self.max_text_len, self.system_prompt)
        return batch_features, batch_lengths, input_ids, labels


class SeqModelAlignEvalDataCollator(object):
    def __init__(self, tokenizer, conf: DataConfig, max_len: int, max_text_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_text_len = max_text_len
        self.names = conf.feature_names

    def __call__(self, examples: List[Dict[str, np.ndarray]]):
        batch_features = {}
        batch_lengths = []
        for feat_name in self.names:
            feats = []
            for e in examples:
                feat = e[feat_name][-self.max_len:]
                if feat_name == "uni_seq_item_id":
                    batch_lengths.append(len(feat))
                feat = feat + [[] for _ in range(self.max_len - len(feat))]
                feats.append(feat)

            values, offests1, _ = row_to_column_2d(feats)
            batch_features[feat_name] = (
                torch.from_numpy(values),
                torch.from_numpy(offests1),
            )
        batch_lengths = torch.tensor(batch_lengths, dtype=torch.long)

        inputs = []
        questions = []
        answers = []
        for e in examples:
            message = json.loads(e["text"])
            input_chatml = message[:1]
            question = message[0]["content"]
            answer = message[1]["content"]
            inputs.append(input_chatml)
            questions.append(question)
            answers.append(answer)
        
        encoded = self.tokenizer.apply_chat_template(
            inputs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            padding=False,
            truncation=True,
            return_tensors="pt",
            # max_length=self.max_text_len,
        )
        input_ids = encoded["input_ids"]
        return batch_features, batch_lengths, input_ids, questions, answers
