from dataclasses import dataclass, field
from transformers import HfArgumentParser
import json
import pickle
import glob
import tqdm
import tvm
from meta_common import register_data_path
import os
import shutil


@dataclass
class ScriptArguments:
    target: str = field(metadata={"help": ""})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    register_data_path(script_args.target)
    script_args.target = tvm.target.Target(script_args.target)

    from meta_common import MEASURE_RECORD_FOLDER, clean_name
    assert(MEASURE_RECORD_FOLDER is not None)

    dirs = glob.glob(f'{MEASURE_RECORD_FOLDER}/*')
    for dir in dirs:
        shutil.rmtree(dir)
    dirs = []
    from meta_utils import get_finetuning_dirs, get_testtuning_dirs
    dirs.extend(get_finetuning_dirs())
    dirs.extend(get_testtuning_dirs())

    # measured_record_set = set()
    database_dic = {}
    for dir in tqdm.tqdm(dirs):
        database_dirs = glob.glob(f'{dir}/*')
        for work_dir in database_dirs:
            task_name = os.path.basename(work_dir)
            path_tuning_record = os.path.join(work_dir, 'database_tuning_record.json')
            path_workload = os.path.join(work_dir, 'database_workload.json')
            if os.path.exists(path_workload) is False or os.path.exists(path_tuning_record) is False:
                continue
            with open(path_workload, 'r') as f:
                workload_str = f.read()
            with open(path_tuning_record, 'r') as f:
                records_str = f.read()

            if task_name not in database_dic:
                database_dic[task_name] = [workload_str, records_str]
            else:
                # assert(workload_str == database_dic[task_name][0])
                database_dic[task_name][1] += records_str

    for task_name, (workload_str, records_str) in tqdm.tqdm(database_dic.items()):
        work_dir = f"{MEASURE_RECORD_FOLDER}/{task_name}"
        os.makedirs(work_dir)
        path_tuning_record = os.path.join(work_dir, 'database_tuning_record.json')
        path_workload = os.path.join(work_dir, 'database_workload.json')
        with open(path_workload, 'w') as f:
            f.write(workload_str)
        with open(path_tuning_record, 'w') as f:
            f.write(records_str)

    # from meta_common import HARDWARE_PLATFORM
    # assert HARDWARE_PLATFORM is not None
    # meta_measured_pkl_path = f'meta_measured_{HARDWARE_PLATFORM}.pkl'
    # with open(meta_measured_pkl_path, 'wb') as f:
    #     pickle.dump(measured_record_set, f)


# meta_measured_pkl = None
if __name__ == "__main__":
    main()


# def check_measured(i_str):
#     global meta_measured_pkl
#     if meta_measured_pkl is None:
#         from meta_common import HARDWARE_PLATFORM
#         assert HARDWARE_PLATFORM is not None
#         meta_measured_pkl_path = f'meta_measured_{HARDWARE_PLATFORM}.pkl'
#         with open(meta_measured_pkl_path, 'rb') as f:
#             meta_measured_pkl = pickle.load(f)
#     measured = i_str in meta_measured_pkl
#     # if measured:
#     #     print('measured', end=' ')
#     return measured