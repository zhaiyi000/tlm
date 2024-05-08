import pickle
from tvm import auto_scheduler
import re
import glob


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


def get_relay_ir_filename(target, network_key):
    assert(NETWORK_INFO_FOLDER is not None)
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_key)}.relay.pkl"


def get_task_info_filename(network_key, target):
    assert(NETWORK_INFO_FOLDER is not None)
    network_task_key = (network_key,) + (str(target.kind),)
    return f"{NETWORK_INFO_FOLDER}/{clean_name(network_task_key)}.task.pkl"


def load_tasks_path(target):
    assert(NETWORK_INFO_FOLDER is not None)
    files = glob.glob(f"{NETWORK_INFO_FOLDER}/*{target.kind}*.pkl")
    return files


def load_and_register_tasks():
    assert(NETWORK_INFO_FOLDER is not None)
    tasks = pickle.load(open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "rb"))

    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)

    return tasks


def get_to_measure_filename(task):
    assert(TO_MEASURE_PROGRAM_FOLDER is not None)
    task_key = (task.workload_key, str(task.target.kind))
    return f"{TO_MEASURE_PROGRAM_FOLDER}/{clean_name(task_key)}.json"


def get_measure_record_filename(task, target=None):
    assert(MEASURE_RECORD_FOLDER is not None)
    target = target or task.target
    task_key = (task.workload_key, str(target.kind))
    return f"{MEASURE_RECORD_FOLDER}/{clean_name(task_key)}.json"


def hold_out_task_files(target, only_bert=False):
    if only_bert:
        files = {
            "bert_base": get_task_info_filename(('bert_base', [1,128]), target)
        }
    else:
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


def yield_hold_out_five_files(target, only_bert=False):
    files = hold_out_task_files(target, only_bert=only_bert)

    for workload, file in files.items():
        tasks_part, task_weights = pickle.load(open(file, "rb"))
        for task, weight in zip(tasks_part, task_weights):
            yield workload, task, get_measure_record_filename(task, target), weight


def get_hold_out_five_files(target):
    files = list(set([it[2] for it in list(yield_hold_out_five_files(target))]))
    files.sort()
    return files


def get_bert_files(target):
    files = list(set([it[2] for it in list(yield_hold_out_five_files(target, True))]))
    files.sort()
    return files