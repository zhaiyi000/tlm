from dataclasses import dataclass, field
from transformers import HfArgumentParser, set_seed
import glob
import os
import random
from multiprocessing import Pool
import json
import copy
from tvm import auto_scheduler
from tvm.auto_scheduler.measure_record import load_record_from_string
from common import register_data_path, load_and_register_tasks, get_hold_out_five_files, get_bert_files
import tvm
from functools import partial
import subprocess
import shutil
from tokenizer import train_tokenizer, test_model_max_length
from make_dataset_utils import json_to_token, make_dataset, make_dataset_test
import re
from enum import Enum
from tvm.auto_scheduler.measure import MeasureInput
import numpy as np
import math


FOR_GEN_TOKENIZER = "for_gen_tokenizer"
FOR_LATENCY = "for_latency"
FOR_GEN = "for_gen"
FOR_GEN_BEST = "for_gen_best"
FOR_GEN_EVAL_SKETCH = "for_gen_eval_sketch"
FOR_GEN_EVAL_SKETCH_ONLY_BERT = "for_gen_eval_sketch_only_bert"
FOR_GEN_EVALTUNING_SKETCH = "for_gen_evaltuning_sketch"
FOR_GEN_TRAIN_SKETCH = "for_gen_train_sketch"
FOR_GEN_BEST_ALL = "for_gen_best_all"


@dataclass
class ScriptArguments:
    for_type: str = field(metadata={"help": "", "choices": [FOR_GEN_TOKENIZER, FOR_GEN, FOR_GEN_BEST, FOR_GEN_EVAL_SKETCH, FOR_GEN_TRAIN_SKETCH, FOR_GEN_BEST_ALL, FOR_GEN_EVALTUNING_SKETCH, FOR_LATENCY, FOR_GEN_EVAL_SKETCH_ONLY_BERT]})
    target: str = field(metadata={"help": ""})
    dataset_path: str = field(metadata={"help": ""})
    tokenizer_path: str = field(metadata={"help": ""})

    save_path: str = field(default=None, metadata={"help": ""})
    file_cnt: int = field(default=None, metadata={"help": ""})
    keep_cnt: int = field(default=None, metadata={"help": ""})
    test_file_idx: int = field(default=None, metadata={"help": ""})
    schedule_file_path: str = field(default=None, metadata={"help": ""})


def for_clm_or_mlm(for_type):
    if for_type == FOR_GEN_TOKENIZER or for_type == FOR_GEN or \
       for_type == FOR_GEN_BEST or for_type == FOR_GEN_EVAL_SKETCH or \
       for_type == FOR_GEN_TRAIN_SKETCH or for_type == FOR_GEN_BEST_ALL or \
       for_type == FOR_GEN_EVALTUNING_SKETCH or for_type == FOR_GEN_EVAL_SKETCH_ONLY_BERT:
        return "clm"
    elif for_type == FOR_LATENCY:
        return "mlm"
    else:
        assert(False)


def for_gen(lines):
    input, _ = load_record_from_string(lines[0])
    compute_dag = auto_scheduler.measure.recover_measure_input(input).task.compute_dag.print_min()

    data_list = []
    latency_min = 1e10
    for line in lines:
        json_line = json.loads(line)
        workload_key = json_line["i"][0][0]
        json_line["i"][0][0] = json.loads(workload_key)

        steps = json_line["i"][1][1]
        for step_idx, step in enumerate(steps):
            if step[0] == "SP":
                sp_list = step[4]
                for i in range(len(sp_list)):
                    sp_list[i] = 1

        latencies = json_line["r"][0]
        if latencies == [0]:
            latency = 1e10
        else:
            latency = sum(latencies) / len(latencies)
        latency_min = min(latency_min, latency)

        data = {}
        data["latency"] = latency
        data["text"] = [compute_dag, json_line["i"]]
        data_list.append(data)

    for data in data_list:
        data["labels"] = latency_min / data["latency"]

    return data_list


def for_gen_best(lines):
    input, _ = load_record_from_string(lines[0])
    task = auto_scheduler.measure.recover_measure_input(input).task
    compute_dag = task.compute_dag.print_min()
    workload_key = task.workload_key

    ppt_str_min = {}
    latency_min = 1e10
    random.shuffle(lines)
    for line in lines:
        json_line = json.loads(line)
        workload_key = json_line["i"][0][0]
        json_line["i"][0][0] = json.loads(workload_key)

        ppt_str = None
        steps = json_line["i"][1][1]
        for step_idx, step in enumerate(steps):
            if step[0] == "SP":
                sp_list = step[4]
                for i in range(len(sp_list)):
                    sp_list[i] = 1
            if step[0] == "PPT":
                ppt_str = str(json_line["i"][1][1][:step_idx+1])

        latencies = json_line["r"][0]
        if latencies == [0]:
            latency = 1e10
        else:
            latency = sum(latencies) / len(latencies)

        if latency >= 1e10:
            continue

        latency_min = min(latency_min, latency)

        assert(ppt_str is not None)
        if ppt_str not in ppt_str_min or ppt_str_min[ppt_str][0] > latency:
            data = {}
            data["text"] = [compute_dag, json_line["i"]]
            data["latency"] = latency
            data["line"] = line
            ppt_str_min[ppt_str] = (latency, data)

        
    data_list = [it[1] for it in ppt_str_min.values()]
    # if latency_min >= 1e10:
    #     return []
    # if (len(data_list) > 2):
    #     print(len(data_list))
    data_list_new = []
    for data in data_list:
        labels = latency_min / data["latency"]
    #     if labels < 1.0:
    #         continue
        data["labels"] = labels
        data_list_new.append(data)

    data_list_new.sort(key=lambda x: x["labels"], reverse=True)
    from common import HARDWARE_PLATFORM
    if HARDWARE_PLATFORM == 'i7':
        data_list_new = data_list_new[:1]
    elif HARDWARE_PLATFORM == 'v100':
        # data_list_new = data_list_new[:2]
        pass
    else:
        assert(False)

    return data_list_new


def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x))/temperature)
    return e_x / e_x.sum(axis=0)

def for_gen_eval_sketch(lines, keep_cnt, for_type):
    input, _ = load_record_from_string(lines[0])
    task = auto_scheduler.measure.recover_measure_input(input).task
    compute_dag = task.compute_dag.print_min()
    workload_key = task.workload_key

    json_line_dict = {}
    for line in lines:
        json_line = json.loads(line)
        steps = json_line["i"][1][1]
        ppt_idx = None
        for step_idx, step in enumerate(steps):
            if step[0] == "SP":
                sp_list = step[4]
                for i in range(len(sp_list)):
                    sp_list[i] = 1
                continue
            if step[0] == "PPT":
                ppt_idx = step_idx
                break
        assert(ppt_idx is not None)
        json_line["i"][1][1] = steps[:ppt_idx+1]

        # if for_type == FOR_GEN_TRAIN_SKETCH or for_type == FOR_GEN_EVALTUNING_SKETCH:
        costs = json_line["r"][0]
        if costs == [0]:
            latency = 1e10
        else:
            latency = sum(costs) / len(costs)
        # elif for_type == FOR_GEN_EVAL_SKETCH:
        #     latency = 1
        # else:
        #     assert(False)

        json_line_str = str(json_line["i"])
        if json_line_str not in json_line_dict:
            json_line_dict[json_line_str] = [latency, json_line]
        else:
            json_line_dict[json_line_str][0] = min(json_line_dict[json_line_str][0], latency)

    latency_list, json_line_list = zip(*json_line_dict.values())
    latency_min = min(latency_list)
    latency_list = [latency_min / it for it in latency_list]

    probs = softmax(np.array(latency_list), temperature=0.3)
    indices = np.random.choice(np.arange(len(latency_list)), size=keep_cnt, replace=True, p=probs)

    data_list = []
    for select_i in indices:
        json_line = json_line_list[select_i]
        # json_line["labels"] = latency_list[select_i]
        # json_line["latency"] = latency_min / json_line["labels"]
        data_list.append(json_line)

    return data_list


def input_to_tokens(task, states, input):
    compute_dag = task.compute_dag.print_min()
    json_line_i = json.loads(input.to_json())
    workload_key = json_line_i[0][0]
    json_line_i[0][0] = json.loads(workload_key)

    data_list = []
    for state in states:
        inp = MeasureInput(task, state)
        steps = json.loads(inp.to_json())[1][1]
        for step_idx, step in enumerate(steps):
            if step[0] == "SP":
                sp_list = step[4]
                for i in range(len(sp_list)):
                    sp_list[i] = 1
        json_line_i[1][1] = steps
        data = {}
        data["text"] = [compute_dag, copy.deepcopy(json_line_i)]
        data_list.append(data)

    return [item["text"] for item in json_to_token(data_list)]


def process_file(args, tmp_folder, for_type, keep_cnt):
    file_i, file = args
    print(file_i, end="    \r", flush=True)
    with open(file, "r") as f:
        lines = f.read().strip().split("\n")
    if for_type == FOR_GEN_TOKENIZER or for_type == FOR_GEN or for_type == FOR_LATENCY:
        data_list = for_gen(lines)
        data_list = json_to_token(data_list)
    elif for_type == FOR_GEN_BEST or for_type == FOR_GEN_BEST_ALL:
        data_list = for_gen_best(lines)
        data_list = json_to_token(data_list)
    elif for_type == FOR_GEN_EVAL_SKETCH or for_type == FOR_GEN_TRAIN_SKETCH or for_type == FOR_GEN_EVALTUNING_SKETCH or for_type == FOR_GEN_EVAL_SKETCH_ONLY_BERT:
        data_list = for_gen_eval_sketch(lines, keep_cnt, for_type)
    else:
        assert(False)

    with open(f"{tmp_folder}/{file_i}_part", "w") as f:
        for data in data_list:
            json.dump(data, f)
            f.write("\n")


def token_files_and_merge(for_type, files, save_path, keep_cnt=None):
    os.makedirs(save_path, exist_ok=True)
    filename = f"{save_path}/0_merge.json"
    tmp_folder = f"{save_path}/0_tmp"
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    with Pool(os.cpu_count()) as pool:
        pool.map(partial(process_file, tmp_folder=tmp_folder, for_type=for_type, keep_cnt=keep_cnt), enumerate(files))
    print()
    subprocess.run(f"cat {tmp_folder}/*_part > {filename}", shell=True)
    shutil.rmtree(tmp_folder)
    return filename


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    # Load task registry
    print("Load all tasks...")
    register_data_path(script_args.target)
    script_args.target = tvm.target.Target(script_args.target)
    tasks = load_and_register_tasks()

    if script_args.for_type == FOR_GEN_TOKENIZER:
        files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
        files.sort()
        print("Dataset file cnt:", len(files))
        if script_args.file_cnt:
            set_seed(0)
            files = random.sample(files, script_args.file_cnt)
            print("Sampled file cnt:", len(files))
        filename = token_files_and_merge(script_args.for_type, files, script_args.tokenizer_path)
        train_tokenizer([filename], script_args.tokenizer_path, test_length=True)
    elif script_args.for_type == FOR_GEN:
        files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
        files.sort()
        print("Dataset file cnt:", len(files))
        hold_out_files = get_hold_out_five_files(script_args.target)
        for out in hold_out_files:
            for file in files:
                if os.path.basename(out) == os.path.basename(file):
                    files.remove(file)
        print("After hold out, file cnt:", len(files))
        if script_args.file_cnt:
            set_seed(0)
            files = random.sample(files, script_args.file_cnt)
            print("Sampled file cnt:", len(files))
        filename = token_files_and_merge(script_args.for_type, files, script_args.save_path)
        make_dataset(filename, script_args.save_path, script_args.tokenizer_path, for_clm_or_mlm(script_args.for_type))
    elif script_args.for_type == FOR_LATENCY:
        files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
        files.sort()
        print("Dataset file cnt:", len(files))
        hold_out_files = get_hold_out_five_files(script_args.target)
        for out in hold_out_files:
            for file in files:
                if os.path.basename(out) == os.path.basename(file):
                    files.remove(file)
        print("After hold out, file cnt:", len(files))
        if script_args.file_cnt:
            set_seed(0)
            files = random.sample(files, script_args.file_cnt)
            print("Sampled file cnt:", len(files))
        filename = token_files_and_merge(script_args.for_type, files, script_args.save_path)
        make_dataset(filename, script_args.save_path, script_args.tokenizer_path, for_clm_or_mlm(script_args.for_type))
    elif script_args.for_type == FOR_GEN_BEST:
        files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
        files.sort()
        print("Dataset file cnt:", len(files))
        hold_out_files = get_hold_out_five_files(script_args.target)
        for out in hold_out_files:
            for file in files:
                if os.path.basename(out) == os.path.basename(file):
                    files.remove(file)
        print("After hold out, file cnt:", len(files))
        if script_args.file_cnt:
            set_seed(0)
            files = random.sample(files, script_args.file_cnt)
            print("Sampled file cnt:", len(files))
        filename = token_files_and_merge(script_args.for_type, files, script_args.save_path)
        make_dataset(filename, script_args.save_path, script_args.tokenizer_path, for_clm_or_mlm(script_args.for_type), valid_percentage=0)
    elif script_args.for_type == FOR_GEN_BEST_ALL:
        files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
        files.sort()
        print("Dataset file cnt:", len(files))
        filename = token_files_and_merge(script_args.for_type, files, script_args.save_path)
        make_dataset(filename, script_args.save_path, script_args.tokenizer_path, for_clm_or_mlm(script_args.for_type), valid_percentage=0)
    elif script_args.for_type == FOR_GEN_EVAL_SKETCH or script_args.for_type == FOR_GEN_EVALTUNING_SKETCH or script_args.for_type == FOR_GEN_EVAL_SKETCH_ONLY_BERT:
        files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
        files.sort()
        print("Dataset file cnt:", len(files))
        if script_args.for_type == FOR_GEN_EVAL_SKETCH_ONLY_BERT:
            hold_out_files = get_bert_files(script_args.target)
        else:
            hold_out_files = get_hold_out_five_files(script_args.target)
        hold_out_set = set()
        for file in hold_out_files:
            hold_out_set.add(os.path.basename(file))
        files_new = []
        for file in files:
            if os.path.basename(file) in hold_out_set:
                files_new.append(file)
        files = files_new
        print("After hold out, file cnt:", len(files))
        if script_args.schedule_file_path:
            from task_sheduler import find_potential_files
            files = find_potential_files(files)
            print("Find potential file cnt:", len(files))
        filename = token_files_and_merge(script_args.for_type, files, script_args.save_path, keep_cnt=script_args.keep_cnt)
    elif script_args.for_type == FOR_GEN_TRAIN_SKETCH:
        files = glob.glob(os.path.join(script_args.dataset_path, "*.json"))
        files.sort()
        print("Dataset file cnt:", len(files))
        hold_out_files = get_hold_out_five_files(script_args.target)
        hold_out_set = set()
        for file in hold_out_files:
            hold_out_set.add(os.path.basename(file))
        files_new = []
        for file in files:
            if os.path.basename(file) not in hold_out_set:
                files_new.append(file)
        files = files_new
        print("After hold out, file cnt:", len(files))
        # if 'to_measure_programs' in files[0]:
        files_new = []
        for file_i, file in enumerate(files):
            if file_i % 4 == script_args.test_file_idx % 4:
                files_new.append(file)
        files = files_new
        print(f"test_file_idx: {script_args.test_file_idx}, len files: {len(files)}")
        # else:
        #     from task_sheduler import find_potential_files
        #     files = find_potential_files(files)
        #     print("Find potential file cnt:", len(files))
        filename = token_files_and_merge(script_args.for_type, files, script_args.save_path, keep_cnt=script_args.keep_cnt)
    else:
        assert(False)


if __name__ == "__main__":
    main()