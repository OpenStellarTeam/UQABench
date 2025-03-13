#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import sys
import json
from collections import OrderedDict


def mean(lst):
    return sum(lst) / len(lst)


def dialogue_evaluation(ori_cands, ori_golds):
    assert len(ori_cands) == len(
        ori_golds), f"num cand: {len(ori_cands)}, num gold: {len(ori_golds)}"
    cnt = 0
    cor = 0
    for cand, gold in zip(ori_cands, ori_golds):
        cnt += 1
        if cand.startswith(gold):
            cor += 1
    result = {"acc": round(100 * cor / cnt, 6)}
    return result


def file_dialogue_evaluation(filename):
    cands = []
    golds = []
    task_dict = OrderedDict({
        "All tasks": ([], []),
        "Sequence understanding": ([], []),
        "Action prediction": ([], []),
        "Interest perception": ([], []),
        "df": ([], []),
        "mf": ([], []),
        "ip": ([], []),
        "ap": ([], []),
        "li": ([], []),
        "si": ([], []),
        "it": ([], []),
    })
    with open(filename, "r") as f:
        for line in f:
            item = json.loads(line)
            question = item["question"]
            cand = item["prediction"]
            gold = item["answer"][0] if isinstance(
                item["answer"], list) else item["answer"]
            if "次浏览的商品的" in question or "次（从1开始计数）浏览的" in question:
                sub_task = "df"
                task = "Sequence understanding"
            elif "长度是多少？" in question or "的商品有多少个" in question:
                sub_task = "mf"
                task = "Sequence understanding"
            elif "下一个可能点击的商品的标题是什么" in question:
                sub_task = "ip"
                task = "Action prediction"
            elif "用户下一个可能点击的商品的" in question:
                sub_task = "ap"
                task = "Action prediction"
            elif "用户最感兴趣的3个" in question:
                sub_task = "li"
                task = "Interest perception"
            elif "用户近期最感兴趣的" in question:
                sub_task = "si"
                task = "Interest perception"
            elif "偏好的转移路径是什么" in question:
                sub_task = "it"
                task = "Interest perception"
            else:
                assert 0, f"question {question} is invalid"

            task_dict[task][0].append(cand)
            task_dict[task][1].append(gold)

            task_dict[sub_task][0].append(cand)
            task_dict[sub_task][1].append(gold)

            task_dict["All tasks"][0].append(cand)
            task_dict["All tasks"][1].append(gold)
    print(f"num samples to eval: {len(task_dict['All tasks'][0])}")
    all_results = {}
    for task_name, (cands, golds) in task_dict.items():
        results = dialogue_evaluation(cands, golds)
        print(f"===== {task_name} =====")
        print(results)
        all_results[task_name] = results
    return all_results


if __name__ == "__main__":
    filename = sys.argv[1]
    file_dialogue_evaluation(filename)
