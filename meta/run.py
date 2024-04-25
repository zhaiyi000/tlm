from dataclasses import dataclass, field
from transformers import HfArgumentParser
import subprocess, os, glob, time, shutil
from urllib.parse import quote
from meta_common import register_data_path
import tvm
from filelock import FileLock
from multiprocessing import Process


FOR_FINETUNING = "for_finetuning"
FOR_TESTTUNING = "for_testtuning"
device_id_all = "6,5,4,3"


@dataclass
class ScriptArguments:
    for_type: str = field(metadata={"help": "", "choices": [FOR_FINETUNING, FOR_TESTTUNING]})
    target: str = field(metadata={"help": ""})

    finetuning_init: bool = field(default=False, metadata={"help": ""})
    testtuning_init: bool = field(default=False, metadata={"help": ""})


def exec_cmd_if_error_send_mail(command):
    print("#" * 50)
    print("command:", command)
    returncode = os.system(command)
    if returncode != 0:
        os.system(f'curl https://diyi.site/ma?text={quote(command)} --noproxy diyi.site')
    return returncode


def run_tuning(for_type, finetuning_init, testtuning_init, ssh_target, model, target, init_times, finetuning_schedule_times, testtuning_schedule_times):
    target_tvm = tvm.target.Target(target)
    if for_type == FOR_FINETUNING and finetuning_init:
        for test_file_idx in range(0, init_times, 1):
            command = f'''
                python meta_make_dataset.py \
                --for_type=for_gen_train_sketch \
                --target="{target}" \
                --dataset_path=dataset/to_measure_programs/{model} \
                --tokenizer_path=meta_data/{model}_tokenizer \
                --save_path=meta_data/{model}_gen_train \
                --keep_cnt=48 \
                --test_file_idx={test_file_idx}
            '''
            exec_cmd_if_error_send_mail(command)

            command = f'''
                CUDA_VISIBLE_DEVICES={device_id_all} python meta_gen_state.py \
                --model_name_or_path=meta_data/clm_gen_{model} \
                --sketch_path=meta_data/{model}_gen_train/0_merge.json \
                --save_path=meta_data/{model}_gen_train/finetuning_{test_file_idx} \
                --allow_repeat=True \
                --target="{target}" \
                --keep_cnt=16
            '''
            exec_cmd_if_error_send_mail(command)

            command = f'cd meta_data/{model}_gen_train; zip -q -r finetuning_{test_file_idx}.zip finetuning_{test_file_idx}/'
            exec_cmd_if_error_send_mail(command)
            command = f'rsync meta_data/{model}_gen_train/finetuning_{test_file_idx}.zip {ssh_target}:~/tlm/meta/measure_data/to_measure/finetuning_{test_file_idx}.zip'
            exec_cmd_if_error_send_mail(command)
            command = f'rm meta_data/{model}_gen_train/finetuning_{test_file_idx}.zip'
            exec_cmd_if_error_send_mail(command)
            command = f'ssh {ssh_target} "cd ~/tlm/meta/measure_data/to_measure; unzip -q -o finetuning_{test_file_idx}.zip; rm finetuning_{test_file_idx}.zip"'
            exec_cmd_if_error_send_mail(command)

    if for_type == FOR_TESTTUNING and testtuning_init:
        for test_file_idx in range(0, init_times, 1):
            command = f'''
                python meta_make_dataset.py \
                --for_type=for_gen_evaltuning_sketch \
                --target="{target}" \
                --dataset_path=dataset/to_measure_programs/{model} \
                --tokenizer_path=meta_data/{model}_tokenizer \
                --save_path=meta_data/{model}_gen_evaltuning \
                --keep_cnt=64
            '''
            exec_cmd_if_error_send_mail(command)

            command = f'''
                CUDA_VISIBLE_DEVICES={device_id_all} python meta_gen_state.py \
                --model_name_or_path=meta_data/clm_gen_best_{model} \
                --sketch_path=meta_data/{model}_gen_evaltuning/0_merge.json \
                --save_path=meta_data/{model}_gen_evaltuning/testtuning_{test_file_idx}  \
                --allow_repeat=True \
                --target="{target}" \
                --keep_cnt=32
            '''
            exec_cmd_if_error_send_mail(command)

            command = f'cd meta_data/{model}_gen_evaltuning; zip -q -r testtuning_{test_file_idx}.zip testtuning_{test_file_idx}/'
            exec_cmd_if_error_send_mail(command)
            command = f'rsync meta_data/{model}_gen_evaltuning/testtuning_{test_file_idx}.zip {ssh_target}:~/tlm/meta/measure_data/to_measure/testtuning_{test_file_idx}.zip'
            exec_cmd_if_error_send_mail(command)
            command = f'rm meta_data/{model}_gen_evaltuning/testtuning_{test_file_idx}.zip'
            exec_cmd_if_error_send_mail(command)
            command = f'ssh {ssh_target} "cd ~/tlm/meta/measure_data/to_measure; unzip -q -o testtuning_{test_file_idx}.zip; rm testtuning_{test_file_idx}.zip"'
            exec_cmd_if_error_send_mail(command)

    # if for_type == FOR_FINETUNING:
    if True:
        want_idx = None
        while True:
            lock = FileLock("../gen/my_lock.lock")
            with lock:
                command = f'ssh {ssh_target} "cd ~/tlm/meta/measure_data/measured; ls -v"'
                result = subprocess.run(command, capture_output=True, shell=True, text=True)
                measured_files = [file for file in result.stdout.strip().split('\n') if file]
                if len(measured_files) > 0:
                    file_path = measured_files[0]
                    if "finetuning_" in file_path:
                        command = f'ssh {ssh_target} "cd ~/tlm/meta/measure_data/measured; zip -q -r {file_path}.zip {file_path}/"'
                        exec_cmd_if_error_send_mail(command)
                        command = f'rsync {ssh_target}:~/tlm/meta/measure_data/measured/{file_path}.zip measured/{file_path}.zip'
                        exec_cmd_if_error_send_mail(command)
                        command = f'cd measured; unzip -q -o {file_path}.zip; rm {file_path}.zip'
                        exec_cmd_if_error_send_mail(command)
                        command = f'ssh {ssh_target} "cd ~/tlm/meta/measure_data/measured; rm {file_path}.zip"'
                        exec_cmd_if_error_send_mail(command)

                        test_file_idx = int(os.path.splitext(file_path)[0][len("finetuning_"):])
                        print(f"Fine tuning file: {file_path}")

                        if want_idx is None:
                            want_idx = test_file_idx
                        if test_file_idx != want_idx:
                            time.sleep(10)
                            continue
                        want_idx += 1

                        command = f'rm -rf meta_data/measure_data_{model}/{file_path}; mv measured/{file_path} meta_data/measure_data_{model}/{file_path}'
                        exec_cmd_if_error_send_mail(command)

                        from meta_utils import add_finetuning_dirs
                        add_finetuning_dirs(f"meta_data/measure_data_{model}/{file_path}")

                        exec_cmd_if_error_send_mail(f'python meta_postprocess.py --target="{target}"')

                        command = f'''
                            python meta_make_dataset.py \
                            --for_type=for_gen_best \
                            --target="{target}" \
                            --dataset_path=dataset/measure_records/{model} \
                            --tokenizer_path=meta_data/{model}_tokenizer \
                            --save_path=meta_data/{model}_gen_best
                        '''
                        exec_cmd_if_error_send_mail(command)
                        
                        exec_cmd_if_error_send_mail(f'python run_train_clm_best_{model}.py')

                        if test_file_idx + init_times < finetuning_schedule_times:
                            command = f'''
                                python meta_make_dataset.py \
                                --for_type=for_gen_train_sketch \
                                --target="{target}" \
                                --dataset_path=dataset/to_measure_programs/{model} \
                                --tokenizer_path=meta_data/{model}_tokenizer \
                                --save_path=meta_data/{model}_gen_train \
                                --keep_cnt=48 \
                                --test_file_idx={test_file_idx + init_times}
                            '''
                            exec_cmd_if_error_send_mail(command)
                        else:
                            # exec_cmd_if_error_send_mail(f'python task_sheduler.py --target="{target}"')
                            # command = f'''
                            #     python make_dataset.py \
                            #     --for_type=for_gen_train_sketch \
                            #     --target="{target}" \
                            #     --dataset_path=dataset/measure_records/{model} \
                            #     --tokenizer_path=gen_data/gen_tokenizer_{model} \
                            #     --save_path=gen_data/{model}_gen_train \
                            #     --keep_cnt=48 \
                            #     --test_file_idx={test_file_idx + init_times}
                            # '''
                            # exec_cmd_if_error_send_mail(command)
                            assert(False)

                        while True:
                            result = subprocess.run('tmux ls', capture_output=True, shell=True, text=True).stdout
                            if f'run_train_clm_best_{model}_py' in result:
                                print(f"run_train_clm_best_{model}_py is running...")
                                time.sleep(10)
                            else:
                                break

                        # if test_file_idx >= (finetuning_schedule_times - 1) and test_file_idx % 4 == 3:
                        if test_file_idx % 4 == 3:
                            command = f'''
                                CUDA_VISIBLE_DEVICES={device_id_all} python meta_gen_state.py \
                                --model_name_or_path=meta_data/clm_gen_best_{model} \
                                --sketch_path=meta_data/{model}_gen_eval/0_merge.json \
                                --save_path=meta_data/{model}_gen_eval/0_test_{test_file_idx} \
                                --allow_repeat=True \
                                --target="{target}" \
                                --keep_cnt=32
                            '''
                            exec_cmd_if_error_send_mail(command)

                            command = f'cd meta_data/{model}_gen_eval; zip -q -r 0_test_{test_file_idx}.zip 0_test_{test_file_idx}/'
                            exec_cmd_if_error_send_mail(command)
                            command = f'rsync meta_data/{model}_gen_eval/0_test_{test_file_idx}.zip {ssh_target}:~/tlm/meta/measure_data/to_measure/0_test_{test_file_idx}.zip'
                            exec_cmd_if_error_send_mail(command)
                            command = f'rm meta_data/{model}_gen_eval/0_test_{test_file_idx}.zip'
                            exec_cmd_if_error_send_mail(command)
                            command = f'ssh {ssh_target} "cd ~/tlm/meta/measure_data/to_measure; unzip -q -o 0_test_{test_file_idx}.zip; rm 0_test_{test_file_idx}.zip"'
                            exec_cmd_if_error_send_mail(command)

                        command = f'''
                            CUDA_VISIBLE_DEVICES={device_id_all} python meta_gen_state.py \
                            --model_name_or_path=meta_data/clm_gen_best_{model} \
                            --sketch_path=meta_data/{model}_gen_train/0_merge.json \
                            --save_path=meta_data/{model}_gen_train/finetuning_{test_file_idx + init_times} \
                            --allow_repeat=False \
                            --target="{target}" \
                            --keep_cnt=16
                        '''
                        exec_cmd_if_error_send_mail(command)

                        command = f'cd meta_data/{model}_gen_train; zip -q -r finetuning_{test_file_idx + init_times}.zip finetuning_{test_file_idx + init_times}/'
                        exec_cmd_if_error_send_mail(command)
                        command = f'rsync meta_data/{model}_gen_train/finetuning_{test_file_idx + init_times}.zip {ssh_target}:~/tlm/meta/measure_data/to_measure/finetuning_{test_file_idx + init_times}.zip'
                        exec_cmd_if_error_send_mail(command)
                        command = f'rm meta_data/{model}_gen_train/finetuning_{test_file_idx + init_times}.zip'
                        exec_cmd_if_error_send_mail(command)
                        command = f'ssh {ssh_target} "cd ~/tlm/meta/measure_data/to_measure; unzip -q -o finetuning_{test_file_idx + init_times}.zip; rm finetuning_{test_file_idx + init_times}.zip"'
                        exec_cmd_if_error_send_mail(command)

                        # exec_cmd_if_error_send_mail(f'rm clm_gen_best_{model}/*bin')

                        command = f'ssh {ssh_target} "cd tlm/meta/measure_data/measured; rm -rf ../moved/{file_path}; mv {file_path} ../moved/{file_path}"'
                        exec_cmd_if_error_send_mail(command)

                    elif "0_test_" in file_path:
                        command = f'ssh {ssh_target} "cd ~/tlm/meta/measure_data/measured; zip -q -r {file_path}.zip {file_path}/"'
                        exec_cmd_if_error_send_mail(command)
                        command = f'rsync {ssh_target}:~/tlm/meta/measure_data/measured/{file_path}.zip measured/{file_path}.zip'
                        exec_cmd_if_error_send_mail(command)
                        command = f'cd measured; unzip -q -o {file_path}.zip; rm {file_path}.zip'
                        exec_cmd_if_error_send_mail(command)
                        command = f'ssh {ssh_target} "cd ~/tlm/meta/measure_data/measured; rm {file_path}.zip"'
                        exec_cmd_if_error_send_mail(command)

                        test_file_idx = int(os.path.splitext(file_path)[0][len("0_test_"):])
                        print(f"Test file: {file_path}")

                        command = f'rm -rf meta_data/measure_data_{model}/{file_path}; mv measured/{file_path} meta_data/measure_data_{model}/{file_path}'
                        exec_cmd_if_error_send_mail(command)
                        
                        from meta_utils import add_test_dirs
                        add_test_dirs(f"meta_data/measure_data_{model}/{file_path}")
                        
                        command = f'python meta_speedup_eval.py --target="{target}" --for_test=True >> meta_speedup_eval_{target_tvm.kind.name}.log 2>&1'
                        exec_cmd_if_error_send_mail(command)

                        command = f'curl https://diyi.site/ma?text=0_test_done --noproxy diyi.site'
                        exec_cmd_if_error_send_mail(command)

                        command = f'ssh {ssh_target} "cd tlm/meta/measure_data/measured; rm -rf ../moved/{file_path}; mv {file_path} ../moved/{file_path}"'
                        exec_cmd_if_error_send_mail(command)
                    elif "testtuning_" in file_path:
                        command = f'ssh {ssh_target} "cd ~/tlm/meta/measure_data/measured; zip -q -r {file_path}.zip {file_path}/"'
                        exec_cmd_if_error_send_mail(command)
                        command = f'rsync {ssh_target}:~/tlm/meta/measure_data/measured/{file_path}.zip measured/{file_path}.zip'
                        exec_cmd_if_error_send_mail(command)
                        command = f'cd measured; unzip -q -o {file_path}.zip; rm {file_path}.zip'
                        exec_cmd_if_error_send_mail(command)
                        command = f'ssh {ssh_target} "cd ~/tlm/meta/measure_data/measured; rm {file_path}.zip"'
                        exec_cmd_if_error_send_mail(command)

                        test_file_idx = int(os.path.splitext(file_path)[0][len("testtuning_"):])
                        print(f"Test tuning file: {file_path}")

                        if want_idx is None:
                            want_idx = test_file_idx
                        if test_file_idx != want_idx:
                            time.sleep(10)
                            continue
                        want_idx += 1

                        command = f'rm -rf meta_data/measure_data_{model}/{file_path}; mv measured/{file_path} meta_data/measure_data_{model}/{file_path}'
                        exec_cmd_if_error_send_mail(command)

                        from meta_utils import add_testtuning_dirs
                        add_testtuning_dirs(f"meta_data/measure_data_{model}/{file_path}")

                        exec_cmd_if_error_send_mail(f'python meta_postprocess.py --target="{target}"')

                        command = f'''
                            python meta_make_dataset.py \
                            --for_type=for_gen_best_all \
                            --target="{target}" \
                            --dataset_path=dataset/measure_records/{model} \
                            --tokenizer_path=meta_data/{model}_tokenizer \
                            --save_path=meta_data/{model}_gen_best
                        '''
                        exec_cmd_if_error_send_mail(command)

                        exec_cmd_if_error_send_mail(f'python run_train_clm_best_{model}.py')

                        exec_cmd_if_error_send_mail(f'python meta_task_sheduler.py --target="{target}" --for_testtuning=True')

                        keep_cnt_factor = 1
                        from meta_task_sheduler import find_potential_dirs_len
                        len_tasks = find_potential_dirs_len()
                        if len_tasks == 0:
                            print('testtuning done...')
                            return
                        if len_tasks <= 30:
                            keep_cnt_factor = 4
                        elif len_tasks <= 40:
                            keep_cnt_factor = 3
                        elif len_tasks <= 60:
                            keep_cnt_factor = 2
                        
                        
                        from meta_common import HARDWARE_PLATFORM
                        if test_file_idx + init_times < testtuning_schedule_times:
                            command = f'''
                                python meta_make_dataset.py \
                                --for_type=for_gen_evaltuning_sketch \
                                --target="{target}" \
                                --dataset_path=dataset/to_measure_programs/{model} \
                                --tokenizer_path=meta_data/{model}_tokenizer \
                                --save_path=meta_data/{model}_gen_evaltuning \
                                --keep_cnt={128*keep_cnt_factor} \
                                --schedule_file_path=meta_task_sheduler_{HARDWARE_PLATFORM}.pkl
                            '''
                            exec_cmd_if_error_send_mail(command)
                        else:
                            assert(False)
                            command = f'''
                                python make_dataset.py \
                                --for_type=for_gen_evaltuning_sketch \
                                --target="{target}" \
                                --dataset_path=dataset/measure_records/{model} \
                                --tokenizer_path=gen_data/gen_tokenizer_{model} \
                                --save_path=gen_data/{model}_gen_evaltuning \
                                --keep_cnt={128*keep_cnt_factor} \
                                --schedule_file_path=task_sheduler_{HARDWARE_PLATFORM}.pkl
                            '''
                            exec_cmd_if_error_send_mail(command)

                        while True:
                            result = subprocess.run('tmux ls', capture_output=True, shell=True, text=True).stdout
                            if f'run_train_clm_best_{model}_py' in result:
                                print(f"run_train_clm_best_{model}_py is running...")
                                time.sleep(10)
                            else:
                                break

                        command = f'''
                            CUDA_VISIBLE_DEVICES={device_id_all} python meta_gen_state.py \
                            --model_name_or_path=meta_data/clm_gen_best_{model} \
                            --sketch_path=meta_data/{model}_gen_evaltuning/0_merge.json \
                            --save_path=meta_data/{model}_gen_evaltuning/testtuning_{test_file_idx + init_times} \
                            --allow_repeat=False \
                            --target="{target}" \
                            --keep_cnt={64*keep_cnt_factor}
                        '''
                        exec_cmd_if_error_send_mail(command)

                        command = f'cd meta_data/{model}_gen_evaltuning; zip -q -r testtuning_{test_file_idx + init_times}.zip testtuning_{test_file_idx + init_times}/'
                        exec_cmd_if_error_send_mail(command)
                        command = f'rsync meta_data/{model}_gen_evaltuning/testtuning_{test_file_idx + init_times}.zip {ssh_target}:~/tlm/meta/measure_data/to_measure/testtuning_{test_file_idx + init_times}.zip'
                        exec_cmd_if_error_send_mail(command)
                        command = f'rm meta_data/{model}_gen_evaltuning/testtuning_{test_file_idx + init_times}.zip'
                        exec_cmd_if_error_send_mail(command)
                        command = f'ssh {ssh_target} "cd ~/tlm/meta/measure_data/to_measure; unzip -q -o testtuning_{test_file_idx + init_times}.zip; rm testtuning_{test_file_idx + init_times}.zip"'
                        exec_cmd_if_error_send_mail(command)

                        command = f'python meta_speedup_eval.py --target="{target}" --for_testtuning=True >> meta_speedup_eval_{target_tvm.kind.name}_testtuning.log 2>&1'
                        exec_cmd_if_error_send_mail(command)

                        command = f'ssh {ssh_target} "cd tlm/meta/measure_data/measured; rm -rf ../moved/{file_path}; mv {file_path} ../moved/{file_path}"'
                        exec_cmd_if_error_send_mail(command)
                    else:
                        print(f"Invalid file: {file_path}")

            print("sleep...")
            time.sleep(10)


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    register_data_path(script_args.target)

    from meta_common import HARDWARE_PLATFORM

    os.makedirs('measured', exist_ok=True)
    os.makedirs(f'meta_data/measure_data_{HARDWARE_PLATFORM}', exist_ok=True)

    if HARDWARE_PLATFORM == 'i7':
        ssh_target = 'hw'
        init_times = 2
        finetuning_schedule_times = 1e10
        testtuning_schedule_times = 1e10
    elif HARDWARE_PLATFORM == 'v100':
        ssh_target = 'zy27'
        init_times = 2
        finetuning_schedule_times = 1e10
        testtuning_schedule_times = 1e10
    elif HARDWARE_PLATFORM == '2080':
        ssh_target = 'zy36'
        init_times = 2
        finetuning_schedule_times = 1e10
        testtuning_schedule_times = 1e10
    else:
        assert(False)
    run_tuning(script_args.for_type, script_args.finetuning_init, script_args.testtuning_init, ssh_target, HARDWARE_PLATFORM, script_args.target, init_times, finetuning_schedule_times, testtuning_schedule_times)


if __name__ == "__main__":
    main()

# PYTHONUNBUFFERED=1 python run.py --target="llvm -mcpu=core-avx2 -model=i7 -num-cores=4" --for_type=for_finetuning --finetuning_init=False |& tee run_i7.log

# PYTHONUNBUFFERED=1 python run.py --target="nvidia/nvidia-v100" --for_type=for_finetuning --finetuning_init=False |& tee run_v100.log