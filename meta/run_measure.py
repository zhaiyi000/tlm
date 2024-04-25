import os, glob, time
from multiprocessing import Process, Lock
from urllib.parse import quote
import math
import shutil


n_part = 12


def exec_cmd_if_error_send_mail(command):
    print("#" * 50)
    print("command:", command)
    returncode = os.system(command)
    if returncode != 0:
        os.system(f'curl https://diyi.site/ma?text={quote(command)} --noproxy diyi.site')
    return returncode


def divide_worker(lock):
    while True:
        to_measure_list = glob.glob("measure_data/to_measure/*")
        to_measure_list = [x for x in to_measure_list if '_part_' not in x and 'zip' not in x]
        to_measure_list.sort()
        if len(to_measure_list) > 0:
            with lock:
                time.sleep(10)
                to_measure_dir = to_measure_list[0]
                tasks = glob.glob(f'{to_measure_dir}/*')
                part_len = math.ceil(len(tasks) / n_part)
                for i in range(n_part):
                    dir_i = f"{to_measure_dir}_part_{i}"
                    os.makedirs(dir_i)
                    tasks_part = tasks[i*part_len : (i+1)*part_len]
                    for task in tasks_part:
                        source_dir = task
                        target_dir = source_dir.replace(to_measure_dir, dir_i)
                        shutil.copytree(source_dir, target_dir)
                to_measure_bak_dir = os.path.join("measure_data/to_measure_bak", os.path.basename(to_measure_dir))
                command = f"rm -rf {to_measure_bak_dir}; mv {to_measure_dir} {to_measure_bak_dir}"
                exec_cmd_if_error_send_mail(command)
        time.sleep(10)


def merge_worker(lock):
    while True:
        measure_part_list = glob.glob("measure_data/measured_part/*_part_*")
        for part_0 in measure_part_list:
            if '_part_0' in part_0:
                finish = True
                merge_dir_list = [part_0]
                for i in range(1, n_part, 1):
                    part_i = f'{part_0[:-1]}{i}'
                    if part_i not in measure_part_list:
                        finish = False
                        break
                    merge_dir_list.append(part_i)
                if finish:
                    with lock:
                        measured_part_dir = os.path.join("measure_data/measured_part", os.path.basename(part_0[:-len('_part_0')]))
                        for dir in merge_dir_list:
                            tasks = glob.glob(f'{dir}/*')
                            for task in tasks:
                                src = task
                                dest = task.replace(dir, measured_part_dir)
                                shutil.move(src, dest)
                            shutil.rmtree(dir)
                        measured_dir = os.path.join("measure_data/measured", os.path.basename(part_0[:-len('_part_0')]))
                        src = measured_part_dir
                        dest = measured_dir
                        shutil.move(src, dest)
        time.sleep(10)


def worker(gpu_id, lock):
    time.sleep(4 - gpu_id)

    while True:
        with lock:
            to_measure_list = glob.glob("measure_data/to_measure/*_part_*")
            to_measure_list.sort()
            to_measure_file = None
            if len(to_measure_list) > 0:
                to_measure_file = to_measure_list[0]

                to_measure_dir = f"measure_data/to_measure_{gpu_id}"
                os.makedirs(to_measure_dir, exist_ok=True)
                to_measure_dir_file = os.path.join(to_measure_dir, os.path.basename(to_measure_file))
                exec_cmd_if_error_send_mail(f"mv {to_measure_file} {to_measure_dir_file}")

        if to_measure_file:
            measured_tmp_file = os.path.join("measure_data/measured_tmp", os.path.basename(to_measure_file))
            command = f"CUDA_VISIBLE_DEVICES={gpu_id} python measure_programs.py --target=\"nvidia/nvidia-v100\" --candidate_cache_dir={to_measure_dir_file} --result_cache_dir={measured_tmp_file} > run_{gpu_id}.log 2>&1"
            exec_cmd_if_error_send_mail(command)
            time.sleep(3)

            # to_measure_bak_file = os.path.join("measure_data/to_measure_bak", os.path.basename(to_measure_file))
            command = f"rm -rf {to_measure_dir_file}"
            exec_cmd_if_error_send_mail(command)

            measured_file = os.path.join("measure_data/measured_part", os.path.basename(to_measure_file))
            with lock:
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

os.system(f'mv measure_data/to_measure_*/*_part_* measure_data/to_measure/')

try:
    available_ids = [3,2,1]
    processes = []

    lock = Lock()

    p = Process(target=divide_worker, args=(lock, ))
    p.start()
    processes.append(p)

    p = Process(target=merge_worker, args=(lock, ))
    p.start()
    processes.append(p)

    for id in available_ids:
        p = Process(target=worker, args=(id, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
except:
    print("Received KeyboardInterrupt, terminating workers")
    for p in processes:
        p.terminate()

# PYTHONUNBUFFERED=1 python run_measure.py |& tee run.log