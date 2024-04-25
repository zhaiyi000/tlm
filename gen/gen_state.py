from dataclasses import dataclass, field
from transformers import HfArgumentParser
from tvm import auto_scheduler
from common import register_data_path, load_and_register_tasks
import tvm
from make_dataset import input_to_tokens
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tqdm
import time
import os
import random
import json
from postprocess import check_measured
import math
from multiprocessing import Process, Queue
import subprocess
import shutil


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


def gen_func(task, states, input, tokenizer, model, device, gen_kwargs):
    if len(states) == 0:
        return []
    tokens = input_to_tokens(task, states, input)
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer(tokens, padding=True, max_length=None)
    except Exception as e:
        print(e)
        print(task, states, input, tokenizer, model, device, gen_kwargs)
        raise Exception()
    input_ids_all = batch["input_ids"]
    attention_mask_all = batch["attention_mask"]
    batch_size = 64

    response_list = []
    with torch.no_grad():
        for start in range(0, len(input_ids_all), batch_size):
            input_ids = input_ids_all[start : start + batch_size]
            attention_mask = attention_mask_all[start : start + batch_size]

            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)[:, :-1]
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)[:, :-1]
            gen_kwargs['max_new_tokens'] = min(gen_kwargs['max_new_tokens'], tokenizer.model_max_length - input_ids.shape[-1])

            response = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            response = response[:, input_ids.shape[-1]:]
            response_list.extend(response.tolist())
    return [tokenizer.batch_decode(item) for item in response_list]


def worker(err_queue, save_path_i, sketch_dic_list_i, gen_kwargs, tokenizer, model_name_or_path, device, allow_repeat, keep_cnt, is_build):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        model.eval()
        builder = auto_scheduler.measure.LocalBuilder(timeout=30)
        if os.path.exists(save_path_i):
            # tag = input(script_args.save_path + ' exist, delete it? [n]')
            # if tag == 'y':
            os.remove(save_path_i)
        for workload_key, inputs in tqdm.tqdm(sketch_dic_list_i):
            def gen_func_inner(task, states, max_new_tokens):
                max_new_tokens = max(max_new_tokens, 1)
                gen_kwargs["max_new_tokens"] = max_new_tokens
                return gen_func(task, states, inputs[0], tokenizer, model, device, gen_kwargs)

            policy = auto_scheduler.SketchPolicy(inputs[0].task)
            measure_inputs = []
            measure_results = []
            input_set = set()
            
            # from filelock import FileLock
            # lock = FileLock('/root/tlm/gen/my_lock.lock')
            # with lock:
            retry_i = 0
            while retry_i < 5:
                all_state_list = policy.gen_states([inp.state for inp in inputs], gen_func_inner)
                # measure_inputs_cnt_before = len(measure_inputs)

                measure_inputs_tmp = []
                for state in all_state_list:
                    inp = auto_scheduler.MeasureInput(inputs[0].task, state)
                    i_str = inp.to_json()
                    if i_str in input_set:
                        continue
                    if allow_repeat is False and check_measured(i_str):
                        continue
                    
                    input_set.add(i_str)
                    measure_inputs_tmp.append(inp)

                default_build_result = auto_scheduler.measure.BuildResult(None, [], 0, None, 0)
                if is_build:
                    build_results = builder.build(measure_inputs_tmp)
                else:
                    build_results = [default_build_result for x in measure_inputs_tmp]
                for res, inp in zip(build_results, measure_inputs_tmp):
                    if res.error_no == 0:
                        measure_inputs.append(inp)
                        measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))

                retry_i += 1
                # measure_inputs_cnt_after = len(measure_inputs)
                # if measure_inputs_cnt_before == measure_inputs_cnt_after:
                #     retry_i += 1
                # else:
                #     retry_i = 0
                if len(measure_inputs) >= keep_cnt:
                    break
            
            if len(measure_inputs) > keep_cnt:
                measure_inputs, measure_results = zip(
                    *random.sample(list(zip(measure_inputs, measure_results)), keep_cnt)
                )
            auto_scheduler.save_records(save_path_i, measure_inputs, measure_results)
    except Exception as e:
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
    tasks = load_and_register_tasks()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    gen_kwargs = {
        "min_length": -1,
        # "max_length": tokenizer.model_max_length,
        "top_k": 0,
        "top_p": 1,
        "num_return_sequences": 1,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.sep_token_id
    }

    inputs, _ = auto_scheduler.RecordReader(script_args.sketch_path).read_lines()
    sketch_dic = {}
    inp_dic = {}
    for inp in tqdm.tqdm(inputs):
        workload_key = inp.task.workload_key
        inp_str = inp.to_json()
        if inp_str in inp_dic:
            inp = auto_scheduler.measure.recover_measure_input(inp_dic[inp_str])
        else:
            inp = auto_scheduler.measure.recover_measure_input(inp, rebuild_state=True)
            inp_dic[inp_str] = inp
        if workload_key not in sketch_dic:
            sketch_dic[workload_key] = []
        sketch_dic[workload_key].append(inp)

    sketch_dic_list = list(sketch_dic.items())
    num_gpus = torch.cuda.device_count()
    per_len = math.ceil(len(sketch_dic_list) / num_gpus)
    # filelist = []
    processes = []
    tmp_folder = '.gen_state'
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    err_queue = Queue()
    for gpu_i in range(num_gpus):
        save_path_i = f'{tmp_folder}/{gpu_i}_part'
        # filelist.append(save_path_i)
        sketch_dic_list_i = sketch_dic_list[gpu_i*per_len : (gpu_i+1)*per_len]
        device = f'cuda:{gpu_i}'
        p = Process(target=worker, args=(err_queue, save_path_i, sketch_dic_list_i, gen_kwargs, tokenizer, script_args.model_name_or_path, device, script_args.allow_repeat, script_args.keep_cnt, script_args.is_build))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    if not err_queue.empty():
        raise Exception(f"An exception occurred in the child process: {err_queue.get()}")


    subprocess.run(f"cat {tmp_folder}/*_part > {script_args.save_path}", shell=True)
    shutil.rmtree(tmp_folder)
    


if __name__ == "__main__":
    main()