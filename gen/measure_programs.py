"""Measure all programs

Usage:
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=e5-2666"
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=e5-2673"
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=epyc-7452"
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=epyc-7r32"
python3 measure_programs.py --target "llvm -mcpu=core-avx2 -model=i7-8750h"
python3 measure_programs.py --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m"
python3 measure_programs.py --target "llvm -mcpu=skylake-avx512 -model=platinum-8272l"
python3 measure_programs.py --target "llvm -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod -model=graviton2"
python3 measure_programs.py --target "llvm -mtriple=aarch64-linux-gnu -mattr=+neon -model=a72" --other-args "--rpc-device-key rasp4b-64 --rpc-host kraken --rpc-port 9191 --rpc-n-parallel 4"
"""

import argparse
import glob
import os
import pickle
import time

from tqdm import tqdm

import tvm
from tvm import auto_scheduler

from common import (register_data_path, load_and_register_tasks,
    get_measure_record_filename, get_to_measure_filename)
import json
from tvm.auto_scheduler.measure_record import load_record_from_string

def make_measurer(run_timeout, repeat, number, enable_cpu_cache_flush,
                  verbose, log_filename, min_repeat_ms):
    builder = auto_scheduler.measure.LocalBuilder(timeout=30)
    runner = auto_scheduler.measure.LocalRunner(
        timeout=run_timeout, repeat=repeat, number=number,
        enable_cpu_cache_flush=enable_cpu_cache_flush, min_repeat_ms=min_repeat_ms)
    measurer = auto_scheduler.measure.ProgramMeasurer(
	builder,
	runner,
        [auto_scheduler.RecordToFile(log_filename)],
	verbose=verbose,
    )
    return measurer


def remeasure_file(task_idx, inputs, target, target_host, batch_size, measurer_kwargs, measured_path):
    # Make measuer
    measurer_kwargs['log_filename'] = measured_path
    measurer = make_measurer(**measurer_kwargs)

    task = auto_scheduler.measure.recover_measure_input(inputs[0]).task
    task = auto_scheduler.SearchTask(
        workload_key=task.workload_key,
        target=target,
        target_host=target_host,
        hardware_params=task.hardware_params,
        layout_rewrite_option=task.layout_rewrite_option,
    )
    empty_policy = auto_scheduler.search_policy.EmptyPolicy(task)

    # Do measurement
    for i in range(0, len(inputs), batch_size):
        print(f"===== task: {task_idx}\t programs: {i}/{len(inputs)} =====")
        inp_batch = []
        for inp in inputs[i:min(len(inputs), i + batch_size)]:
            inp_batch.append(auto_scheduler.MeasureInput(task, inp.state))
        res_batch = measurer.measure(task, empty_policy, inp_batch)

        timeout_ct = 0
        for res in res_batch:
            if res.error_no == auto_scheduler.measure.MeasureErrorNo.BUILD_TIMEOUT:
                timeout_ct += 1

def main(args):
    measured_set = set()
    if os.path.exists(args.measured_path):
        with open(args.measured_path, 'r') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            inp_str = json.dumps(json.loads(line)['i'])
            measured_set.add(inp_str)

    to_measure_list = []
    with open(args.to_measure_path, 'r') as f:
        lines = f.read().strip().split('\n')
    with open(args.measured_path, 'w') as f:
        pass
    for line in lines:
        if line:
            inp_str = json.dumps(json.loads(line)['i'])
            if inp_str in measured_set:
                continue
            to_measure_list.append(line)
    
    # inputs, _ = auto_scheduler.RecordReader(args.to_measure_path).read_lines()
    input_dict = {}
    for inp_str in to_measure_list:
        inp, _ = load_record_from_string(inp_str)
        task = auto_scheduler.measure.recover_measure_input(inp).task
        if task.workload_key not in input_dict:
            input_dict[task.workload_key] = (task, [])
        input_dict[task.workload_key][1].append(inp)

    end_idx = min(args.end_idx, len(tasks))

    print("len input_dict:", len(input_dict))
    # if os.path.exists(args.measured_path):
    #     os.remove(args.measured_path)

    # Remeasure all tasks
    for task_i, (workload_key, (task, records)) in enumerate(input_dict.items()):
        target = tvm.target.Target(args.target)
        if target.kind.name == 'llvm':
            # Set measurement arguments
            measurer_kwargs = {
                "run_timeout": 5,
                "number": 1,
                "enable_cpu_cache_flush": True,
                "verbose": 1,
                "min_repeat_ms": 100
            }
            if task.compute_dag.flop_ct >= 2416443392.0:
                measurer_kwargs['repeat'] = 4
            elif task.compute_dag.flop_ct >= 834928640.0:
                measurer_kwargs['repeat'] = 6
            elif task.compute_dag.flop_ct <= 2097152.0:
                measurer_kwargs['repeat'] = 10
            else:
                measurer_kwargs['repeat'] = 8
        elif target.kind.name == 'cuda':
            measurer_kwargs = {
                "run_timeout": 5,
                "number": 3,
                "enable_cpu_cache_flush": False,
                "verbose": 1,
                "repeat": 1,
                "min_repeat_ms": 300
            }
        else:
            assert(False)

        # Run measurement
        # task_key = (task.workload_key, str(task.target.kind))

        remeasure_file(task_i, records, target, args.target_host, args.batch_size, measurer_kwargs, args.measured_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="llvm -mcpu=core-avx2 --model=i7")
    parser.add_argument("--target-host", type=str)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=1000000)
    parser.add_argument("--step-idx", type=int, default=1)
    parser.add_argument("--to-measure-path", type=str, required=True)
    parser.add_argument("--measured-path", type=str, required=True)
    args = parser.parse_args()

    # Load task registry
    register_data_path(args.target)
    print("Load all tasks...")
    tasks = load_and_register_tasks()

    main(args)



# python measure_programs.py --target="llvm -mcpu=core-avx2 -model=i7" --to-measure-path=1122.json --measured-path=tmp.json

# python measure_programs.py --target="nvidia/nvidia-v100" --to-measure-path=1122.json --measured-path=tmp.json
