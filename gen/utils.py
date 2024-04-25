import os, json
from common import HARDWARE_PLATFORM


utils_json_path = "utils.json"
utils_json = None


def get_utils_json(key):
    assert HARDWARE_PLATFORM is not None
    global utils_json
    utils_json = {}
    try:
        with open(utils_json_path, 'r') as f:
            utils_json = json.load(f)
    except:
        pass
    if HARDWARE_PLATFORM not in utils_json:
        utils_json[HARDWARE_PLATFORM] = {}
    if key:
       if key not in utils_json[HARDWARE_PLATFORM]:
           utils_json[HARDWARE_PLATFORM][key] = []
    return utils_json[HARDWARE_PLATFORM][key]


def save_utils_json():
    with open(utils_json_path, 'w') as f:
        json.dump(utils_json, f, indent=2)


def add_finetuning_files(file):
    tmp_list = get_utils_json('finetuning_files')
    
    if file not in tmp_list:
        tmp_list.append(file)
    save_utils_json()


def get_finetuning_files():
    tmp_list = get_utils_json('finetuning_files')
    return tmp_list


def add_test_files(file):
    tmp_list = get_utils_json('test_files')

    if file not in tmp_list:
        tmp_list.append(file)
    save_utils_json()


def get_test_files():
    tmp_list = get_utils_json('test_files')
    return tmp_list


def add_testtuning_files(file):
    tmp_list = get_utils_json('testtuning_files')

    if file not in tmp_list:
        tmp_list.append(file)
    save_utils_json()


def get_testtuning_files():
    tmp_list = get_utils_json('testtuning_files')
    return tmp_list

