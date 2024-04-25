from dataclasses import dataclass, field
from transformers import HfArgumentParser
import json
import pickle
import glob
import tqdm
import tvm
from common import register_data_path
import os


@dataclass
class ScriptArguments:
    target: str = field(metadata={"help": ""})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    register_data_path(script_args.target)
    script_args.target = tvm.target.Target(script_args.target)

    from common import MEASURE_RECORD_FOLDER, clean_name
    os.makedirs(MEASURE_RECORD_FOLDER, exist_ok=True)
    assert(MEASURE_RECORD_FOLDER is not None)

    files = glob.glob(f'{MEASURE_RECORD_FOLDER}/*.json')
    for file in files:
        os.remove(file)
    files = []
    from utils import get_finetuning_files, get_testtuning_files
    files.extend(get_finetuning_files())
    files.extend(get_testtuning_files())

    record_dic = {}
    measured_record_set = set()

    for file in tqdm.tqdm(files):
        with open(file, 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                json_line = json.loads(line)
                workload_key = json_line["i"][0][0]
                if workload_key not in record_dic:
                    record_dic[workload_key] = []

                i_str = json.dumps(json.loads(line)['i'])
                if i_str in measured_record_set:
                    continue
                else:
                    measured_record_set.add(i_str)
                    record_dic[workload_key].append(line)

    for workload_key, lines in tqdm.tqdm(record_dic.items()):
        task_key = (workload_key, str(script_args.target.kind))
        filename = f"{MEASURE_RECORD_FOLDER}/{clean_name(task_key)}.json"
        with open(filename, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')

    from common import HARDWARE_PLATFORM
    assert HARDWARE_PLATFORM is not None
    measured_pkl_path = f'measured_{HARDWARE_PLATFORM}.pkl'
    with open(measured_pkl_path, 'wb') as f:
        pickle.dump(measured_record_set, f)


measured_pkl = None
if __name__ == "__main__":
    main()


def check_measured(i_str):
    global measured_pkl
    if measured_pkl is None:
        from common import HARDWARE_PLATFORM
        assert HARDWARE_PLATFORM is not None
        measured_pkl_path = f'measured_{HARDWARE_PLATFORM}.pkl'
        with open(measured_pkl_path, 'rb') as f:
            measured_pkl = pickle.load(f)
    measured = i_str in measured_pkl
    # if measured:
    #     print('measured', end=' ')
    return measured