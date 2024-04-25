from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
import json
from tvm import auto_scheduler
import tvm
import numpy as np
import tqdm
import pickle
import os
from meta_common import register_data_path, hold_out_task_files, get_jsondatabase_top1
import tvm.meta_schedule as ms
from meta_common import yield_hold_out_five_files


@dataclass
class ScriptArguments:
    target: str = field(metadata={"help": ""})

    test_file: str = field(default=None, metadata={"help": ""})
    for_train: str = field(default=False, metadata={"help": ""})
    for_test: str = field(default=False, metadata={"help": ""})
    for_testtuning: str = field(default=False, metadata={"help": ""})
    pass


# def read_gen_best_json(json_path):
#     with open(json_path, "r") as f:
#         lines = f.read().strip().split("\n")
#     min_latency_dict = {}
#     for line in lines:
#         json_line = json.loads(line)
#         latency = json_line["latency"]
#         json_line = json.loads(json_line["note"]["line"])
#         workload_key = json_line["i"][0][0]
#         if workload_key not in min_latency_dict:
#             min_latency_dict[workload_key] = 1e10
#         min_latency_dict[workload_key] = min(min_latency_dict[workload_key], latency)
#     return min_latency_dict


# def read_fine_tuning_json(json_path):
#     with open(json_path, "r") as f:
#         lines = f.read().strip().split("\n")
#     min_latency_dict = {}
#     for line in lines:
#         json_line = json.loads(line)
#         latencies = json_line["r"][0]
#         latency = sum(latencies) / len(latencies)
#         workload_key = json_line["i"][0][0]
#         if workload_key not in min_latency_dict:
#             min_latency_dict[workload_key] = 1e10
#         min_latency_dict[workload_key] = min(min_latency_dict[workload_key], latency)
#     return min_latency_dict


# def best_lines_convert():
#     from common import HARDWARE_PLATFORM
#     with open(f'gen_data/{HARDWARE_PLATFORM}_gen_best/0_merge.json', 'r') as f:
#         lines = f.read().strip().split('\n')

#     target_path = f'gen_data/measure_data_{HARDWARE_PLATFORM}/best.json'
#     with open(target_path, 'w') as f:
#         for line in lines:
#             f.write(json.loads(line)['line'])
#             f.write('\n')
#     return target_path


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    # # Load task registry
    # print("Load all tasks...")
    # tasks = load_and_register_tasks()
    register_data_path(script_args.target)
    script_args.target = tvm.target.Target(script_args.target)

    if script_args.for_train:
        assert(False)
        # test_file = best_lines_convert()
        # from gen_utils import get_finetuning_files
        # print('get_testtuning_files', get_finetuning_files()[-1])
    elif script_args.for_test:
        from meta_utils import get_test_dirs
        test_dir = get_test_dirs()[-1]
    elif script_args.for_testtuning:
        from meta_common import HARDWARE_PLATFORM
        test_dir = f'dataset/measure_records/{HARDWARE_PLATFORM}'
        from meta_utils import get_testtuning_dirs
        print('get_testtuning_dirs', get_testtuning_dirs()[-1])
    elif script_args.test_file:
        assert(False)
        # test_file = script_args.test_file
    else:
        assert(False)
        
    print('-' * 50)
    print("test dir:", test_dir)
    
    workloads_latency = {}
    for workload, task, hash_taskname, task_weight in yield_hold_out_five_files(script_args.target):
        work_dir = f'{test_dir}/{hash_taskname}'
        # database = ms.database.JSONDatabase(work_dir=work_dir)
        # first_record = database.get_all_tuning_records()[0]
        # top1_record = database.get_top_k(first_record.workload, 1)[0]
        min_latency, _ = get_jsondatabase_top1(work_dir)

        if workload not in workloads_latency:
            workloads_latency[workload] = 0
        workloads_latency[workload] += float(min_latency) * task_weight
    
    val_total = 0
    for key, val in workloads_latency.items():
        print(f"{key}: {val * 1000:.4f}")
        val_total += val
    print(f"{val_total * 1000:.4f}")


if __name__ == "__main__":
    main()

