import os, glob, json
import numpy as np
from meta_common import register_data_path, get_hold_out_five_files
import tvm

def get_all_dirs(dataset_path):
    all_files_and_dirs = os.listdir(dataset_path)
    all_dirs = [os.path.join(dataset_path, d) for d in all_files_and_dirs if os.path.isdir(os.path.join(dataset_path, d))]
    return all_dirs

def merge_topk(topk):
    all_dirs = get_all_dirs('/root/tlm/meta/dataset/measure_records/v100')
    all_workload = []
    all_line = []

    register_data_path('nvidia/nvidia-v100')
    hold_out_files_set = set(get_hold_out_five_files(tvm.target.Target('nvidia/nvidia-v100')))

    valid_cnt = 0
    for dir_i, dir in enumerate(all_dirs):

        if os.path.basename(dir) not in hold_out_files_set:
            continue

        path_tuning_record = os.path.join(dir, 'database_tuning_record.json')
        path_workload = os.path.join(dir, 'database_workload.json')


        with open(path_workload, 'r') as f:
            lines = f.read().strip().split('\n')
            assert(len(lines) == 1)
            all_workload.append(lines[0])


        with open(path_tuning_record, 'r') as f:
            lines = f.read().strip().split('\n')[:topk]
            for line in lines:
                assert(line[:3] == '[0,')
                all_line.append(f'[{valid_cnt},' + line[3:])

                json_line = json.loads(line)
                latency = np.mean(json_line[1][1])
                if latency > 1000:
                    import pdb; pdb.set_trace()
        valid_cnt += 1

        
    os.makedirs(f'tlm_top_{topk}', exist_ok=True)
    with open(f'tlm_top_{topk}/database_workload.json', 'w') as f:
        f.write('\n'.join(all_workload))

    with open(f'tlm_top_{topk}/database_tuning_record.json', 'w') as f:
        f.write('\n'.join(all_line))



merge_topk(1)
merge_topk(10)
merge_topk(32)
merge_topk(20000)