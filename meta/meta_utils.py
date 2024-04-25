import os, json
from meta_common import HARDWARE_PLATFORM


meta_utils_json_path = "meta_utils.json"
meta_utils_json = None


def get_meta_utils_json(key):
    assert HARDWARE_PLATFORM is not None
    global meta_utils_json
    meta_utils_json = {}
    try:
        with open(meta_utils_json_path, 'r') as f:
            meta_utils_json = json.load(f)
    except:
        pass
    if HARDWARE_PLATFORM not in meta_utils_json:
        meta_utils_json[HARDWARE_PLATFORM] = {}
    if key:
       if key not in meta_utils_json[HARDWARE_PLATFORM]:
           meta_utils_json[HARDWARE_PLATFORM][key] = []
    return meta_utils_json[HARDWARE_PLATFORM][key]


def save_meta_utils_json():
    with open(meta_utils_json_path, 'w') as f:
        json.dump(meta_utils_json, f, indent=2)


def add_finetuning_dirs(dir):
    tmp_list = get_meta_utils_json('finetuning_dirs')
    
    if dir not in tmp_list:
        tmp_list.append(dir)
    save_meta_utils_json()


def get_finetuning_dirs():
    tmp_list = get_meta_utils_json('finetuning_dirs')
    return tmp_list


def add_test_dirs(dir):
    tmp_list = get_meta_utils_json('test_dirs')

    if dir not in tmp_list:
        tmp_list.append(dir)
    save_meta_utils_json()


def get_test_dirs():
    tmp_list = get_meta_utils_json('test_dirs')
    return tmp_list


def add_testtuning_dirs(dir):
    tmp_list = get_meta_utils_json('testtuning_dirs')

    if dir not in tmp_list:
        tmp_list.append(dir)
    save_meta_utils_json()


def get_testtuning_dirs():
    tmp_list = get_meta_utils_json('testtuning_dirs')
    return tmp_list

