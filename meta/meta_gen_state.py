from dataclasses import dataclass, field
from transformers import HfArgumentParser
from tvm import auto_scheduler
from meta_common import register_data_path, load_hash_tasks, remove_trailing_numbers  # , load_and_register_tasks
import tvm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tqdm
import time
import os
import random
import json
import math
from multiprocessing import Process, Queue
import subprocess
import shutil
import tvm.meta_schedule as ms
import tempfile
from tvm.meta_schedule.logging import get_loggers_from_work_dir
import copy


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": ""})
    sketch_path: str = field(metadata={"help": ""})
    save_path: str = field(metadata={"help": ""})
    keep_cnt: int = field(metadata={"help": ""})
    target: str = field(metadata={"help": ""})

    # device: str = field(default="cuda:0", metadata={"help": ""})
    allow_repeat: bool = field(default=True, metadata={"help": ""})
    is_build: bool = field(default=False, metadata={"help": ""})


# def gen_func(task, states, input, tokenizer, model, device, gen_kwargs):
#     tokens = input_to_tokens(task, states, input)
#     tokenizer.padding_side = "left"
#     batch = tokenizer(tokens, padding=True, max_length=None)
#     input_ids_all = batch["input_ids"]
#     attention_mask_all = batch["attention_mask"]
#     batch_size = 64

#     response_list = []
#     with torch.no_grad():
#         for start in range(0, len(input_ids_all), batch_size):
#             input_ids = input_ids_all[start : start + batch_size]
#             attention_mask = attention_mask_all[start : start + batch_size]

#             input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)[:, :-1]
#             attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)[:, :-1]
#             # gen_kwargs['max_new_tokens'] = min(gen_kwargs['max_new_tokens'], tokenizer.model_max_length - input_ids.shape[-1])

#             response = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
#             response = response[:, input_ids.shape[-1]:]
#             response_list.extend(response.tolist())
#     return [tokenizer.batch_decode(item) for item in response_list]


def worker(err_queue, save_path, sketch_dic_list_i, hash_task_i, dataset_path, gen_kwargs, tokenizer, model_name_or_path, device, allow_repeat, keep_cnt, is_build):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        model.eval()
        for hash, json_lines in tqdm.tqdm(sketch_dic_list_i):
            # def gen_func_inner(task, states, max_new_tokens):
            #     max_new_tokens = max(max_new_tokens, 1)
            #     gen_kwargs["max_new_tokens"] = max_new_tokens
            #     return gen_func(task, states, inputs[0], tokenizer, model, device, gen_kwargs)

            # policy = auto_scheduler.SketchPolicy(inputs[0].task)
            # measure_inputs = []
            # measure_results = []
            # input_set = set()
            task = hash_task_i[hash]
            database_name = f'{hash}__{remove_trailing_numbers(task.task_name)}'
            work_dir = os.path.join(save_path, database_name)
            if os.path.exists(work_dir):
                # shutil.rmtree(work_dir)
                assert(False)
            logger = get_loggers_from_work_dir(work_dir, [task.task_name])[0]
            rand_state = ms.utils.fork_seed(None, n=1)[0]

            database_path = os.path.join(dataset_path, database_name)
            path_workload_src = os.path.join(database_path, 'database_workload.json')
            path_workload_dest = os.path.join(work_dir, 'database_workload.json')
            path_tuning_record = os.path.join(work_dir, 'database_tuning_record.json')
            shutil.copyfile(path_workload_src, path_workload_dest)
            with open(path_tuning_record, 'w') as f:
                for line in json_lines:
                    f.write(line['line'])
                    f.write('\n')
            database = ms.database.JSONDatabase(work_dir=work_dir)
            commit_dir = os.path.join(work_dir, 'commit')
            os.makedirs(commit_dir)
            commit_database = ms.database.JSONDatabase(work_dir=commit_dir)
            commit_idx = 0
            commit_records = []

            def get_measured_str_from_line(line):
                json_line = json.loads(line)
                assert(json_line[0] == 0)
                want_part = copy.deepcopy(json_line[1])
                del want_part[1]
                measured_str = json.dumps(want_part)
                return measured_str

            def get_measured_set():
                if allow_repeat:
                    return set()
                from meta_common import HARDWARE_PLATFORM
                dataset_path = f'dataset/measure_records/{HARDWARE_PLATFORM}'
                database_path = os.path.join(dataset_path, database_name)
                path_tuning_record = os.path.join(database_path, 'database_tuning_record.json')
                if os.path.exists(path_tuning_record) is False:
                    return set()
                with open(path_tuning_record, 'r') as f:
                    lines = [x for x in f.read().strip().split('\n') if x]
                measured_set = set()
                for line in lines:
                    measured_set.add(get_measured_str_from_line(line))
                return measured_set
            measured_set = get_measured_set()

            def update_measured_set(measured_set, commit_records, commit_dir):
                nonlocal commit_idx
                path_tuning_record = os.path.join(commit_dir, 'database_tuning_record.json')
                with open(path_tuning_record, 'r') as f:
                    lines = [x for x in f.read().strip().split('\n') if x]
                lines = lines[commit_idx:]
                for line in lines:
                    measured_str = get_measured_str_from_line(line)
                    if measured_str not in measured_set:
                        measured_set.add(measured_str)
                        commit_records.append(line)
                commit_idx += len(lines)

            ctx = ms.TuneContext(
                mod=task.dispatched[0],
                target=task.target,
                space_generator='post-order-apply',
                search_strategy=ms.search_strategy.EvolutionarySearch(population_size=4000),
                task_name=task.task_name,
                logger=logger,
                rand_state=rand_state,
                num_threads='physical'
            ).clone()
            
            tokenizer.padding_side = "left"
            batch = tokenizer([line['text'] for line in json_lines], padding=True, max_length=None)
            input_ids_all = batch["input_ids"]
            attention_mask_all = batch["attention_mask"]
            batch_size = 16

            retry_i = 0
            while retry_i < 5:
                response_list = []
                with torch.no_grad():
                    for start in range(0, len(input_ids_all), batch_size):
                        input_ids = input_ids_all[start : start + batch_size]
                        attention_mask = attention_mask_all[start : start + batch_size]

                        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)[:, :-1]
                        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)[:, :-1]
                        # gen_kwargs['max_new_tokens'] = min(gen_kwargs['max_new_tokens'], tokenizer.model_max_length - input_ids.shape[-1])

                        response = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
                        response = response[:, input_ids.shape[-1]:]
                        response_list.extend(response.tolist())
                
                decision_tokens = [tokenizer.batch_decode(item) for item in response_list]
                ms.tune.gen_state(
                    tasks=[ctx],
                    task_weights=[task.weight],
                    work_dir=work_dir,
                    max_trials_global=2048,
                    decision_tokens=decision_tokens,
                    database=database,
                    commit_database=commit_database,
                    builder=ms.builder.LocalBuilder(timeout_sec=30),
                    is_build=is_build
                )

                # commit_records_cnt_before = len(commit_records)
                update_measured_set(measured_set, commit_records, commit_dir)
                # commit_records_cnt_after = len(commit_records)
                # if commit_records_cnt_before == commit_records_cnt_after:
                #     retry_i += 1
                # else:
                #     retry_i = 0
                retry_i += 1
                if len(commit_records) >= keep_cnt:
                    break

            if len(commit_records) > keep_cnt:
                commit_records = random.sample(commit_records, keep_cnt)
            with open(path_tuning_record, 'w') as f:
                for line in commit_records:
                    f.write(line)
                    f.write('\n')
            
            shutil.rmtree(os.path.join(work_dir, 'logs'))
            shutil.rmtree(os.path.join(work_dir, 'commit'))

    except Exception as e:
        print('error######!!', e)
        err_queue.put(e)


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    # Load task registry
    print("Load all tasks...")
    register_data_path(script_args.target)
    script_args.target = tvm.target.Target(script_args.target)
    # tasks = load_and_register_tasks()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    gen_kwargs = {
        "min_length": -1,
        "max_length": tokenizer.model_max_length,
        "top_k": 0,
        "top_p": 1,
        "num_return_sequences": 1,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.sep_token_id
    }

    # inputs, _ = auto_scheduler.RecordReader(script_args.sketch_path).read_lines()
    # sketch_dic = {}
    # inp_dic = {}
    # for inp in tqdm.tqdm(inputs):
    #     workload_key = inp.task.workload_key
    #     inp_str = inp.to_json()
    #     if inp_str in inp_dic:
    #         inp = auto_scheduler.measure.recover_measure_input(inp_dic[inp_str])
    #     else:
    #         inp = auto_scheduler.measure.recover_measure_input(inp, rebuild_state=True)
    #         inp_dic[inp_str] = inp
    #     if workload_key not in sketch_dic:
    #         sketch_dic[workload_key] = []
    #     sketch_dic[workload_key].append(inp)
    sketch_dic = {}
    with open(script_args.sketch_path, 'r') as f:
        lines = f.read().strip().split('\n')
        for line in tqdm.tqdm(lines):
            json_line = json.loads(line)
            if json_line['hash'] not in sketch_dic:
                sketch_dic[json_line['hash']] = []
            sketch_dic[json_line['hash']].append(json_line)
    sketch_dic_list = list(sketch_dic.items())
    hash_tasks = load_hash_tasks(script_args.target)

    num_gpus = torch.cuda.device_count()
    parallel_cnt = num_gpus * 3
    per_len = math.ceil(len(sketch_dic_list) / parallel_cnt)
    # filelist = []
    processes = []
    err_queue = Queue()
    if os.path.exists(script_args.save_path):
        shutil.rmtree(script_args.save_path)
    os.makedirs(script_args.save_path)
    from meta_common import HARDWARE_PLATFORM
    dataset_path = f'dataset/to_measure_programs/{HARDWARE_PLATFORM}'
    for parallel_i in range(parallel_cnt):
        device = f'cuda:{parallel_i % num_gpus}'
        sketch_dic_list_i = sketch_dic_list[parallel_i*per_len : (parallel_i+1)*per_len]
        hash_task_i = {hash: hash_tasks[hash] for hash, _ in sketch_dic_list_i}

        # worker(err_queue, script_args.save_path, sketch_dic_list_i, hash_task_i, dataset_path, gen_kwargs, tokenizer, script_args.model_name_or_path, device, script_args.allow_repeat, script_args.keep_cnt)
        p = Process(target=worker, args=(err_queue, script_args.save_path, sketch_dic_list_i, hash_task_i, dataset_path, gen_kwargs, tokenizer, script_args.model_name_or_path, device, script_args.allow_repeat, script_args.keep_cnt, script_args.is_build))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    if not err_queue.empty():
        raise Exception(f"An exception occurred in the child process: {err_queue.get()}")
    

if __name__ == "__main__":
    main()