from dataclasses import dataclass, field
from transformers import HfArgumentParser, set_seed
import os, json, glob
import tvm.meta_schedule as ms
import copy
from make_dataset_utils import json_to_token, make_dataset
import tqdm
from multiprocessing import Pool
from tokenizer import train_tokenizer
from meta_common import register_data_path, get_hold_out_five_files
import tvm
from functools import partial
import numpy as np
import shutil
import subprocess
import random


FOR_GEN_TOKENIZER = "for_gen_tokenizer"
FOR_GEN = "for_gen"
FOR_GEN_BEST = "for_gen_best"
FOR_GEN_BEST_ALL = "for_gen_best_all"
FOR_GEN_TRAIN_SKETCH = "for_gen_train_sketch"
FOR_GEN_EVAL_SKETCH = "for_gen_eval_sketch"
FOR_GEN_EVALTUNING_SKETCH = "for_gen_evaltuning_sketch"


@dataclass
class ScriptArguments:
    for_type: str = field(metadata={"help": "", "choices": [FOR_GEN_TOKENIZER, FOR_GEN, FOR_GEN_BEST, FOR_GEN_TRAIN_SKETCH, FOR_GEN_EVAL_SKETCH, FOR_GEN_EVALTUNING_SKETCH, FOR_GEN_BEST_ALL]})
    target: str = field(metadata={"help": ""})
    dataset_path: str = field(metadata={"help": ""})
    tokenizer_path: str = field(metadata={"help": ""})

    save_path: str = field(default=None, metadata={"help": ""})
    file_cnt: int = field(default=None, metadata={"help": ""})
    keep_cnt: int = field(default=None, metadata={"help": ""})
    test_file_idx: int = field(default=None, metadata={"help": ""})
    schedule_file_path: str = field(default=None, metadata={"help": ""})


def recursion_reset_json(json):
    assert(isinstance(json, (list, tuple)))
    for idx, it in enumerate(json):
        if isinstance(it, (list, tuple)):
            recursion_reset_json(it)
        elif isinstance(it, int):
            json[idx] = 1
        else:
            assert(False)


def for_init_workload(work_dir):
    path_tuning_record = os.path.join(work_dir, 'database_tuning_record.json')
    path_workload = os.path.join(work_dir, 'database_workload.json')
    path_tuning_record_1 = os.path.join(work_dir, 'database_tuning_record_1.json')

    with open(path_tuning_record, 'r') as f:
        lines = f.read().strip().split('\n')
    with open(path_tuning_record_1, 'w') as f:
        f.write(lines[0])

    database = ms.database.JSONDatabase(path_workload=path_workload, path_tuning_record=path_tuning_record_1)
    all_records = database.get_all_tuning_records()

    task_name_part = None
    shape_part = []
    target_part = None

    hash_task_name = os.path.basename(work_dir)
    assert('__' in hash_task_name)
    hash, task_name = hash_task_name.split('__')
    task_name_part = task_name.split('_')

    for rec in all_records:
        shape_list = ms.arg_info.ArgInfo.from_entry_func(rec.workload.mod, False)
        for shape in shape_list:
            shape_part.append([shape.dtype, tuple(shape.shape)])
        target_part = str(rec.target)
    
    return lines, path_tuning_record, path_workload, task_name_part, shape_part, target_part, hash, task_name


def for_init_lines(lines, path_tuning_record):
    lines = [x for x in lines if x]
    for line in lines:
        if line == '':
            print('unexpected path_tuning_record', path_tuning_record)
        json_line = json.loads(line)

        insts_part = None
        decisions_part = []
        decisions_label = None
        parallel_label = []

        insts_part, decisions_label = json_line[1][0]
        latency = np.mean(json_line[1][1])
        for inst_i, inst in enumerate(insts_part):
            if inst[0] == 'EnterPostproc':
                break
        insts_part = insts_part[:inst_i+1]
        for inst in insts_part:
            if inst[0] == 'Annotate' and (inst[2] == ['meta_schedule.parallel']):
                parallel_label.append(copy.deepcopy(inst))
                inst[1][1] = 1
        decisions_part = copy.deepcopy(decisions_label)
        recursion_reset_json(decisions_part)
        for dec_i in range(len(decisions_part)):
            dec = decisions_part[dec_i]
            dec_label = decisions_label[dec_i]
            dec[0] = dec_label[0]
        
        yield line, insts_part, decisions_part, decisions_label, parallel_label, latency

    
def for_gen_tokenizer(work_dir):
    data_list = []
    lines, path_tuning_record, path_workload, task_name_part, shape_part, target_part, hash, task_name = for_init_workload(work_dir)
    for line, insts_part, decisions_part, decisions_label, parallel_label, latency in for_init_lines(lines, path_tuning_record):
        ppt = 'PPT'  
        data_list.append({'text': [task_name_part, shape_part, target_part, insts_part, decisions_part,
                                    ppt, 
                                    decisions_label, parallel_label]})
    return data_list


def for_gen_train_sketch(work_dir, keep_cnt):
    prompt_set = set()
    prompt_lines = []
    lines, path_tuning_record, path_workload, task_name_part, shape_part, target_part, hash, task_name = for_init_workload(work_dir)
    for line, insts_part, decisions_part, decisions_label, parallel_label, latency in for_init_lines(lines, path_tuning_record):
        ppt = 'PPT'  
        ppt_line = {'text': [task_name_part, shape_part, target_part, insts_part, decisions_part, ppt],
                    'hash': hash,
                    'task_name': task_name}
        ppt_line_str = str(ppt_line)
        ppt_line['line'] = line
        if ppt_line_str not in prompt_set:
            prompt_set.add(ppt_line_str)
            prompt_lines.append(ppt_line)
    indices = np.random.choice(np.arange(len(prompt_lines)), size=keep_cnt, replace=True)
    data_list = []
    for select_i in indices:
        json_line = prompt_lines[select_i]
        data_list.append(json_line)
    return data_list


def for_gen_best(work_dir):
    prompt_dic = {}
    lines, path_tuning_record, path_workload, task_name_part, shape_part, target_part, hash, task_name = for_init_workload(work_dir)
    random.shuffle(lines)
    min_latency = 1e10
    for line, insts_part, decisions_part, decisions_label, parallel_label, latency in for_init_lines(lines, path_tuning_record):
        min_latency = min(min_latency, latency)
        ppt = 'PPT'
        data_line = {'text': [task_name_part, shape_part, target_part, insts_part, decisions_part,
                                    ppt, 
                                    decisions_label, parallel_label],
                    'latency': latency}
        ppt_line = {'text': [task_name_part, shape_part, target_part, insts_part, decisions_part, ppt],
                    'hash': hash,
                    'task_name': task_name}
        ppt_line_str = str(ppt_line)

        if ppt_line_str not in prompt_dic or prompt_dic[ppt_line_str][0] > latency:
            prompt_dic[ppt_line_str] = (latency, data_line)

    for _, (latency, data_line) in prompt_dic.items():
        data_line['label'] = min_latency / data_line['latency']
        
    prompt_dic_list = [x[1] for x in list(prompt_dic.values())]
    prompt_dic_list.sort(key=lambda x: x['label'], reverse=True)
    from meta_common import HARDWARE_PLATFORM
    if HARDWARE_PLATFORM == 'i7':
        prompt_dic_list = prompt_dic_list[:1]
    elif HARDWARE_PLATFORM == 'v100':
        prompt_dic_list = prompt_dic_list[:1]
    else:
        assert(False)
    return prompt_dic_list


def process_file(args, tmp_folder, for_type, keep_cnt):
    work_dir_i, work_dir = args
    print('work_dir:', work_dir_i, ' ' * 30, end='\r')
    if for_type == FOR_GEN_TOKENIZER or for_type == FOR_GEN:
        data_list = for_gen_tokenizer(work_dir)
        data_list = json_to_token(data_list)
    elif for_type == FOR_GEN_TRAIN_SKETCH or for_type == FOR_GEN_EVAL_SKETCH or for_type == FOR_GEN_EVALTUNING_SKETCH:
        data_list = for_gen_train_sketch(work_dir, keep_cnt)
        data_list = json_to_token(data_list)
    elif for_type == FOR_GEN_BEST or for_type == FOR_GEN_BEST_ALL:
        data_list = for_gen_best(work_dir)
        data_list = json_to_token(data_list)
    else:
        assert(False)

    with open(f"{tmp_folder}/{work_dir_i}_part", "w") as f:
        for data in data_list:
            json.dump(data, f)
            f.write("\n")


def token_files_and_merge(for_type, dirs, save_path, keep_cnt=None):
    os.makedirs(save_path, exist_ok=True)
    filename = f"{save_path}/0_merge.json"
    tmp_folder = f"{save_path}/0_tmp"
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    with Pool() as pool:
        pool.map(partial(process_file, tmp_folder=tmp_folder, for_type=for_type, keep_cnt=keep_cnt), enumerate(dirs))
    print()
    subprocess.run(f"cat {tmp_folder}/*_part > {filename}", shell=True)
    shutil.rmtree(tmp_folder)
    return filename


def get_all_dirs(dataset_path):
    all_files_and_dirs = os.listdir(dataset_path)
    all_dirs = [os.path.join(dataset_path, d) for d in all_files_and_dirs if os.path.isdir(os.path.join(dataset_path, d))]
    return all_dirs


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    register_data_path(script_args.target)
    target = tvm.target.Target(script_args.target)

    if script_args.for_type == FOR_GEN_TOKENIZER:
        all_dirs = []
        all_dirs.extend(get_all_dirs(script_args.dataset_path))
        print('len all dirs:', len(all_dirs))
        filename = token_files_and_merge(script_args.for_type, all_dirs, script_args.tokenizer_path)
        train_tokenizer([filename], script_args.tokenizer_path, test_length=True)
    elif script_args.for_type == FOR_GEN:
        all_dirs = []
        all_dirs.extend(get_all_dirs(script_args.dataset_path))
        print('len all dirs:', len(all_dirs))
        hold_out_files_set = set(get_hold_out_five_files(target))
        all_dirs_new = []
        for dir in all_dirs:
            if os.path.basename(dir) not in hold_out_files_set:
                all_dirs_new.append(dir)
        all_dirs = all_dirs_new
        print('after hold out, len all dirs:', len(all_dirs))
        if script_args.file_cnt:
            set_seed(0)
            all_dirs = random.sample(all_dirs, script_args.file_cnt)
            print("Sampled dir cnt:", len(all_dirs))
        filename = token_files_and_merge(script_args.for_type, all_dirs, script_args.save_path)
        make_dataset(filename, script_args.save_path, script_args.tokenizer_path, 'clm')
    elif script_args.for_type == FOR_GEN_TRAIN_SKETCH:
        all_dirs = []
        all_dirs.extend(get_all_dirs(script_args.dataset_path))
        print('len all dirs:', len(all_dirs))
        hold_out_files_set = set(get_hold_out_five_files(target))
        all_dirs_new = []
        for dir in all_dirs:
            if os.path.basename(dir) not in hold_out_files_set:
                all_dirs_new.append(dir)
        all_dirs = all_dirs_new
        print('after hold out, len all dirs:', len(all_dirs))
        all_dirs_new = []
        if script_args.test_file_idx is not None:
            for dir_i, dir in enumerate(all_dirs):
                if dir_i % 4 == script_args.test_file_idx % 4:
                    all_dirs_new.append(dir)
            all_dirs = all_dirs_new
            print(f"test_file_idx: {script_args.test_file_idx}, len all_dirs: {len(all_dirs)}")
        token_files_and_merge(script_args.for_type, all_dirs, script_args.save_path, keep_cnt=script_args.keep_cnt)
    elif script_args.for_type == FOR_GEN_EVAL_SKETCH or script_args.for_type == FOR_GEN_EVALTUNING_SKETCH:
        all_dirs = []
        all_dirs.extend(get_all_dirs(script_args.dataset_path))
        print('len all dirs:', len(all_dirs))
        hold_out_files_set = set(get_hold_out_five_files(target))
        all_dirs_new = []
        for dir in all_dirs:
            if os.path.basename(dir) in hold_out_files_set:
                all_dirs_new.append(dir)
        all_dirs = all_dirs_new
        print('after hold out, len all dirs:', len(all_dirs))
        if script_args.schedule_file_path:
            from meta_task_sheduler import find_potential_dirs
            all_dirs = find_potential_dirs(all_dirs)
            print("Find potential dir cnt:", len(all_dirs))
        token_files_and_merge(script_args.for_type, all_dirs, script_args.save_path, keep_cnt=script_args.keep_cnt)
    elif script_args.for_type == FOR_GEN_BEST:
        all_dirs = []
        all_dirs.extend(get_all_dirs(script_args.dataset_path))
        print('len all dirs:', len(all_dirs))
        hold_out_files_set = set(get_hold_out_five_files(target))
        all_dirs_new = []
        for dir in all_dirs:
            if os.path.basename(dir) not in hold_out_files_set:
                all_dirs_new.append(dir)
        all_dirs = all_dirs_new
        print('after hold out, len all dirs:', len(all_dirs))
        filename = token_files_and_merge(script_args.for_type, all_dirs, script_args.save_path)
        make_dataset(filename, script_args.save_path, script_args.tokenizer_path, 'clm', valid_percentage=0)
    elif script_args.for_type == FOR_GEN_BEST_ALL:
        all_dirs = []
        all_dirs.extend(get_all_dirs(script_args.dataset_path))
        print('len all dirs:', len(all_dirs))
        filename = token_files_and_merge(script_args.for_type, all_dirs, script_args.save_path)
        make_dataset(filename, script_args.save_path, script_args.tokenizer_path, 'clm', valid_percentage=0)
    else:
        assert(False)


if __name__ == '__main__':
    main()