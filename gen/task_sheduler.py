from dataclasses import dataclass, field
from transformers import HfArgumentParser
import os
import json
import pickle
from common import register_data_path, load_tasks_path, hold_out_task_files
import tqdm
import tvm
import math


@dataclass
class ScriptArguments:
    target: str = field(metadata={"help": ""})
    for_testtuning: bool = field(default=False, metadata={"help": ""})
    pass


def read_fine_tuning_json(json_path):
    with open(json_path, "r") as f:
        lines = f.read().strip().split("\n")
    min_latency_dict = {}
    for line in lines:
        json_line = json.loads(line)
        latencies = json_line["r"][0]
        latency = sum(latencies) / len(latencies)
        workload_key = json_line["i"][0][0]
        if workload_key not in min_latency_dict:
            min_latency_dict[workload_key] = (1e10, 0)
        min_latency_dict[workload_key] = (min(min_latency_dict[workload_key][0], latency), min_latency_dict[workload_key][1] + 1)
    return min_latency_dict


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    register_data_path(script_args.target)
    script_args.target = tvm.target.Target(script_args.target)

    # filenames = load_tasks_path(script_args.target)
    filenames = list(hold_out_task_files(script_args.target).values())
    # for file in hold_out_files:
    #     filenames.remove(file)
    filenames.sort()

    best_history = {}
    times_dic = {}
    from utils import get_finetuning_files, get_testtuning_files
    finetuning_list = get_finetuning_files()
    if script_args.for_testtuning:
        finetuning_list = get_testtuning_files()
    for file in finetuning_list:
        min_latency_dict = read_fine_tuning_json(file)
        for key, (val, times) in min_latency_dict.items():
            if key not in best_history:
                best_history[key] = []
                times_dic[key] = 0
            best_history[key].append(val)
            times_dic[key] += times

    model_dic = {}
    times_model_dic = {}
    for filename in tqdm.tqdm(filenames):
        model_dic[filename] = {}
        times_model_dic[filename] = 0
        tasks, task_weights = pickle.load(open(filename, "rb"))
        for task, weight in zip(tasks, task_weights):
            workload_key = task.workload_key
            if workload_key in best_history:
                model_dic[filename][workload_key] = (best_history[workload_key], weight)
                times_model_dic[filename] += times_dic[workload_key]
            else:
                assert(False)

    from common import clean_name

    key_str_set_total = set()
    print('times_model_dic:', list(times_model_dic.values())[:10])
    for model, tasks_his_wei in model_dic.items():
        if len(tasks_his_wei) == 0:
            assert(False)
            continue
        key_str_set = set()
        for back_window_size in range(3, 100, 1):
            # back_window_size = 2
            weight_score_dic = {}
            for key, (best_list, weight) in tasks_his_wei.items():
                print('', end='')
                for i in range(1, len(best_list), 1):
                    best_list[i] = min(best_list[i], best_list[i-1])

                for _ in range(len(best_list), back_window_size, 1):
                    best_list.append(0)
                best_list = best_list[-back_window_size:]
                score = best_list[0] - min(best_list)
                weight_score_dic[key] = score * weight

            weight_score_dic_list = list(weight_score_dic.items())
            weight_score_dic_list.sort(key=lambda x: x[1], reverse=True)
            want_cnt = math.ceil(len(weight_score_dic_list) / 4)
            want_cnt = min(want_cnt, math.floor((20000 - times_model_dic[model] - want_cnt * 2 * 64) / 64))
            if want_cnt <= 0:
                break

            want_list = [it for it in weight_score_dic_list if it[1] > 0]

            # print(f'{back_window_size}:')
            for i, (key, score) in enumerate(want_list):
                # if i < 5:
                #     print(score, end='  ')
                key_str_set.add(clean_name((key, script_args.target.kind)))
                if len(key_str_set) == want_cnt:
                    break
            if len(key_str_set) == want_cnt:
                    break
        key_str_set_total.update(key_str_set)

    from common import HARDWARE_PLATFORM
    with open(os.path.join(model_path, f'task_sheduler_{HARDWARE_PLATFORM}.pkl'), 'wb') as f:
        pickle.dump(key_str_set_total, f)


model_path = 'gen_data/task_sheduler'
os.makedirs(model_path, exist_ok=True)
if __name__ == "__main__":
    main()
    # find_potential_files()


def find_potential_files(files):
    from common import HARDWARE_PLATFORM
    file_path = os.path.join(model_path, f'task_sheduler_{HARDWARE_PLATFORM}.pkl')
    with open(file_path, 'rb') as f:
        key_str_set = pickle.load(f)
    os.remove(file_path)

    potential_files = []
    for file in files:
        if os.path.splitext(os.path.basename(file))[0] in key_str_set:
            potential_files.append(file)
    return potential_files


def find_potential_files_len():
    from common import HARDWARE_PLATFORM
    file_path = os.path.join(model_path, f'task_sheduler_{HARDWARE_PLATFORM}.pkl')
    with open(file_path, 'rb') as f:
        key_str_set = pickle.load(f)
    return len(key_str_set)