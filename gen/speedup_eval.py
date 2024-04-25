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
from common import register_data_path, hold_out_task_files


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


def best_lines_convert():
    from common import HARDWARE_PLATFORM
    with open(f'gen_data/{HARDWARE_PLATFORM}_gen_best/0_merge.json', 'r') as f:
        lines = f.read().strip().split('\n')

    target_path = f'gen_data/measure_data_{HARDWARE_PLATFORM}/best.json'
    with open(target_path, 'w') as f:
        for line in lines:
            f.write(json.loads(line)['line'])
            f.write('\n')
    return target_path


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    # # Load task registry
    # print("Load all tasks...")
    # tasks = load_and_register_tasks()
    register_data_path(script_args.target)
    script_args.target = tvm.target.Target(script_args.target)
    
    workloads = hold_out_task_files(script_args.target).values()

    if script_args.for_train:
        test_file = best_lines_convert()
        from utils import get_finetuning_files
        print('get_testtuning_files', get_finetuning_files()[-1])
    elif script_args.for_test:
        from utils import get_test_files
        test_file = get_test_files()[-1]
    elif script_args.for_testtuning:
        test_file = best_lines_convert()
        from utils import get_testtuning_files
        print('get_testtuning_files', get_testtuning_files()[-1])
    elif script_args.test_file:
        test_file = script_args.test_file
    else:
        assert(False)
        
    print('-' * 50)
    print("test file:", test_file)
    inputs, results = auto_scheduler.RecordReader(test_file).read_lines()

    input_dict = {}
    for inp, res in zip(inputs, results):
        # task = auto_scheduler.measure.recover_measure_input(inp).task
        if inp.task.workload_key not in input_dict:
            input_dict[inp.task.workload_key] = 1e10
        costs = [x.value for x in res.costs if isinstance(x, tvm.tir.expr.FloatImm)]
        latency = np.mean(costs)
        input_dict[inp.task.workload_key] = min(input_dict[inp.task.workload_key], latency)

    # target = tvm.target.Target("llvm -mcpu=core-avx2 --model=i7")
    # workloads = glob.glob("dataset/network_info/*llvm).task.pkl")

    workloads_latency = {}
    for workload in tqdm.tqdm(workloads):
        tasks_part, task_weights = pickle.load(open(workload, "rb"))
        workload_lat = 0
        for task, weight in zip(tasks_part, task_weights):
            if task.workload_key not in input_dict:
                print(task.workload_key, "task.workload_key not in input_dict")
                continue
            workload_lat += input_dict[task.workload_key] * weight
        workload_name = os.path.splitext(os.path.splitext(os.path.basename(workload))[0])[0]
        if workload_name not in workloads_latency:
            workloads_latency[workload_name] = 0
        workloads_latency[workload_name] += workload_lat
    
    val_total = 0
    for key, val in workloads_latency.items():
        print(f"{key}: {val * 1000:.4f}")
        val_total += val
    print(f"{val_total * 1000:.4f}")


if __name__ == "__main__":
    main()

