import pickle
from tvm import auto_scheduler
import re
import glob
import tempfile
import tvm.meta_schedule as ms
import os
import json


NETWORK_INFO_FOLDER = None
TO_MEASURE_PROGRAM_FOLDER = None
MEASURE_RECORD_FOLDER = None
HARDWARE_PLATFORM = None


def clean_name(x):
    x = str(x)
    x = x.replace(" ", "")
    x = x.replace("\"", "")
    x = x.replace("'", "")
    return x


def get_task_hashes(tasks):
    work_dir = tempfile.TemporaryDirectory()
    database = ms.database.JSONDatabase(work_dir=work_dir.name, module_equality='structural')
    hashes = []
    for task in tasks:
        hashes.append(database.get_hash(task.dispatched[0]))
    work_dir.cleanup()
    return hashes


def remove_trailing_numbers(s):
    last_underscore_index = s.rfind('_')
    if last_underscore_index != -1 and s[last_underscore_index + 1:].isdigit():
        return s[:last_underscore_index]
    return s


def register_data_path(target_str):
    assert(isinstance(target_str, str))
    model_list = ['i7', 'v100', 'a100', '2080', 'None']
    for model in model_list:
        if model in target_str:
            break
    assert(model != 'None')

    print(f'register data path: {model}')
    global NETWORK_INFO_FOLDER, TO_MEASURE_PROGRAM_FOLDER, MEASURE_RECORD_FOLDER, HARDWARE_PLATFORM
    NETWORK_INFO_FOLDER = f"dataset/network_info/{model}"
    TO_MEASURE_PROGRAM_FOLDER = f"dataset/to_measure_programs/{model}"
    MEASURE_RECORD_FOLDER = f"dataset/measure_records/{model}"
    HARDWARE_PLATFORM = model


def get_task_info_filename(network_key, target):
    assert(NETWORK_INFO_FOLDER is not None)
    network_task_key = (network_key,) + (str(target.kind),)
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_task_key)}.task.pkl"


# def get_to_measure_filename(task):
#     assert(TO_MEASURE_PROGRAM_FOLDER is not None)
#     task_key = (get_task_hash(task), str(task.target.kind))
#     return f"{TO_MEASURE_PROGRAM_FOLDER}/{clean_name(task_key)}.json"


# def load_tasks_path(target):
#     assert(NETWORK_INFO_FOLDER is not None)
#     files = glob.glob(f"{NETWORK_INFO_FOLDER}/*{target.kind}*.pkl")
#     return files


def load_tasks():
    assert(NETWORK_INFO_FOLDER is not None)
    tasks = pickle.load(open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "rb"))
    return tasks


def load_hash_tasks(target):
    # cache_pkl = f'{__name__}_{HARDWARE_PLATFORM}_hash_tasks_exclude_hold_out.pkl'
    # if os.path.exists(cache_pkl):
    #     with open(cache_pkl, 'rb') as f:
    #         return pickle.load(f)
    tasks = load_tasks()
    task_hashes = get_task_hashes(tasks)
    task_dic = {hash: task for hash, task in zip(task_hashes, tasks)}
    
    # hold_out = list(set([it[1] for it in list(yield_hold_out_five_files(target))]))
    # hold_out_hashes = set(get_task_hashes(hold_out))
    # for hash in hold_out_hashes:
    #     if hash in task_dic:
    #         del task_dic[hash]
    # with open(cache_pkl, 'wb') as f:
    #     pickle.dump(task_dic, f)
    return task_dic


# def get_measure_record_filename(task):
#     assert(MEASURE_RECORD_FOLDER is not None)
#     return f"{MEASURE_RECORD_FOLDER}/{get_task_hash(task)}__{task.task_name}"


def hold_out_task_files(target):
    files = {
        "resnet_50": get_task_info_filename(('resnet_50', [1,3,224,224]), target),
        "mobilenet_v2": get_task_info_filename(('mobilenet_v2', [1,3,224,224]), target),
        "resnext_50": get_task_info_filename(('resnext_50', [1,3,224,224]), target),
        "bert_base": get_task_info_filename(('bert_base', [1,128]), target),
        # "gpt2": get_task_info_filename(('gpt2', [1,128]), target),
        # "llama": get_task_info_filename(('llama', [4,256]), target),
        "bert_tiny": get_task_info_filename(('bert_tiny', [1,128]), target),
        
        "densenet_121": get_task_info_filename(('densenet_121', [8,3,256,256]), target),
        "bert_large": get_task_info_filename(('bert_large', [4,256]), target),
        "wide_resnet_50": get_task_info_filename(('wide_resnet_50', [8,3,256,256]), target),
        "resnet3d_18": get_task_info_filename(('resnet3d_18', [4,3,144,144,16]), target),
        "dcgan": get_task_info_filename(('dcgan', [8,3,64,64]), target)
    }
    return files


def yield_hold_out_five_files(target):
    files = hold_out_task_files(target)

    for workload, file in files.items():
        tasks_part = pickle.load(open(file, "rb"))
        hashes = get_task_hashes(tasks_part)
        for task, hash in zip(tasks_part, hashes):
            yield workload, task, f"{hash}__{remove_trailing_numbers(task.task_name)}", task.weight


def get_hold_out_five_files(target):
    files = list(set([it[2] for it in list(yield_hold_out_five_files(target))]))
    files.sort()
    return files


def get_jsondatabase_top1(work_dir):
    path_tuning_record = os.path.join(work_dir, 'database_tuning_record.json')
    if os.path.exists(path_tuning_record) is False:
        return 1e10, 0
    with open(path_tuning_record, 'r') as f:
        lines = [x for x in f.read().strip().split('\n') if x]
    min_latency = 1e10
    for line in lines:
        json_line = json.loads(line)
        min_latency = min(min_latency, sum(json_line[1][1]) / len(json_line[1][1]))
    return min_latency, len(lines)