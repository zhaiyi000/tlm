import os, glob, time
from multiprocessing import Process
from urllib.parse import quote
import math


n_part = 6


def exec_cmd_if_error_send_mail(command):
    print("#" * 50)
    print("command:", command)
    returncode = os.system(command)
    if returncode != 0:
        os.system(f'curl https://diyi.site/ma?text={quote(command)} --noproxy diyi.site')
    return returncode


def divide_worker():
    while True:
        to_measure_list = glob.glob("measure_data/to_measure/*.json")
        for file in to_measure_list:
            if '_part_' in file:
                to_measure_list.remove(file)
        to_measure_list.sort()
        if len(to_measure_list) > 0:
            to_measure_file = to_measure_list[0]

            with open(to_measure_file, 'r') as f:
                lines = f.readlines()
            part_len = math.ceil(len(lines) / n_part)
            for i in range(n_part):
                with open(f'{to_measure_file}_part_{i}', 'w') as f:
                    f.writelines(lines[i*part_len : (i+1)*part_len])
            
            to_measure_bak_file = os.path.join("measure_data/to_measure_bak", os.path.basename(to_measure_file))
            command = f"mv {to_measure_file} {to_measure_bak_file}"
            exec_cmd_if_error_send_mail(command)
        time.sleep(10)


def merge_worker():
    while True:
        measure_part_list = glob.glob("measure_data/measured_part/*.json_part_*")
        for part_0 in measure_part_list:
            if '_part_0' in part_0:
                finish = True
                merge_file_list = [part_0]
                for i in range(1, n_part, 1):
                    part_i = f'{part_0[:-1]}{i}'
                    if part_i not in measure_part_list:
                        finish = False
                        break
                    merge_file_list.append(part_i)
                if finish:
                    lines = []
                    for merge_file in merge_file_list:
                        with open(merge_file, 'r') as f:
                            lines.extend(f.readlines())
                    
                    measured_file = os.path.join("measure_data/measured", os.path.basename(part_0[:-len('_part_0')]))
                    with open(measured_file, 'w') as f:
                        f.writelines(lines)
                    for merge_file in merge_file_list:
                        os.remove(merge_file)
        time.sleep(10)


def worker(gpu_id):
    time.sleep(4 - gpu_id)

    while True:
        to_measure_list = glob.glob("measure_data/to_measure/*.json_part_*")
        to_measure_list.sort()
        if len(to_measure_list) > 0:
            to_measure_file = to_measure_list[0]

            to_measure_dir = f"measure_data/to_measure_{gpu_id}"
            os.makedirs(to_measure_dir, exist_ok=True)
            to_measure_dir_file = os.path.join(to_measure_dir, os.path.basename(to_measure_file))

            exec_cmd_if_error_send_mail(f"mv {to_measure_file} {to_measure_dir_file}")

            measured_tmp_file = os.path.join("measure_data/measured_tmp", os.path.basename(to_measure_file))
            command = f"CUDA_VISIBLE_DEVICES={gpu_id} python measure_programs.py --batch-size=64 --target=\"nvidia/nvidia-v100\" --to-measure-path={to_measure_dir_file} --measured-path={measured_tmp_file} > run_{gpu_id}.log 2>&1"
            exec_cmd_if_error_send_mail(command)
            time.sleep(3)

            # to_measure_bak_file = os.path.join("measure_data/to_measure_bak", os.path.basename(to_measure_file))
            command = f"rm {to_measure_dir_file}"
            exec_cmd_if_error_send_mail(command)

            measured_file = os.path.join("measure_data/measured_part", os.path.basename(to_measure_file))
            command = f"mv {measured_tmp_file} {measured_file}"
            exec_cmd_if_error_send_mail(command)
            continue

        print(f"{gpu_id} sleep...")
        time.sleep(10)


os.makedirs('measure_data/to_measure', exist_ok=True)
os.makedirs('measure_data/moved', exist_ok=True)
os.makedirs('measure_data/measured', exist_ok=True)
os.makedirs('measure_data/measured_part', exist_ok=True)
os.makedirs('measure_data/measured_tmp', exist_ok=True)
os.makedirs('measure_data/to_measure_bak', exist_ok=True)

os.system(f'mv measure_data/to_measure_*/*.json_part_* measure_data/to_measure/')

try:
    available_ids = [3, 2, 1]
    processes = []

    p = Process(target=divide_worker, args=())
    p.start()
    processes.append(p)

    p = Process(target=merge_worker, args=())
    p.start()
    processes.append(p)

    for id in available_ids:
        p = Process(target=worker, args=(id,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
except:
    print("Received KeyboardInterrupt, terminating workers")
    for p in processes:
        p.terminate()

# PYTHONUNBUFFERED=1 python run_measure.py |& tee run_measure.log