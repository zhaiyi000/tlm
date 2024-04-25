from tvm import meta_schedule as ms
import pickle
from tvm.meta_schedule.logging import get_loggers_from_work_dir
import os, shutil
import tqdm
import glob
import tempfile
from multiprocessing import Process
import math
from meta_common import register_data_path, load_tasks, get_task_hashes, remove_trailing_numbers
import argparse
import tvm


def worker(part_extracted_tasks, work_i, dir_path):
    hashes = get_task_hashes(part_extracted_tasks)
    for task, hash in tqdm.tqdm(zip(part_extracted_tasks, hashes)):
        work_dir = os.path.join(dir_path, f'{hash}__{remove_trailing_numbers(task.task_name)}')
        if os.path.exists(work_dir):
            with open(os.path.join(work_dir, 'database_tuning_record.json'), 'r') as f:
                lines = [line for line in f.read().strip().split('\n') if line]
                if len(lines) > 0:
                    continue

        logger = get_loggers_from_work_dir(work_dir, [task.task_name])[0]
        rand_state = ms.utils.fork_seed(None, n=1)[0]

        ctx = ms.TuneContext(
            mod=task.dispatched[0],
            target=task.target,
            space_generator='post-order-apply',
            search_strategy=ms.search_strategy.EvolutionarySearch(population_size=1000),
            task_name=task.task_name,
            logger=logger,
            rand_state=rand_state,
            num_threads='physical'
        ).clone()

        ms.tune.dump_program(
            tasks=[ctx],
            task_weights=[task.weight],
            work_dir=work_dir,
            max_trials_global=2048
        )


def main():
    register_data_path(args.target)
    target = tvm.target.Target(args.target)

    all_extracted_tasks = load_tasks()
    print('len extracted tasks:', len(all_extracted_tasks))
    start_idx = args.start_idx or 0
    end_idx = args.end_idx or len(all_extracted_tasks)
    all_extracted_tasks = all_extracted_tasks[start_idx:end_idx]
    
    print('len extracted tasks:', len(all_extracted_tasks))

    from meta_common import HARDWARE_PLATFORM
    dir_path = f'dataset/to_measure_programs/{HARDWARE_PLATFORM}'
    os.makedirs(dir_path, exist_ok=True)

    number_worker = 4
    processes = []
    part_list = [[] for _ in range(number_worker)]

    for task_i, task in enumerate(all_extracted_tasks):
        part_list[task_i % number_worker].append(task)

    for i in range(number_worker):
        p = Process(target=worker, args=(part_list[i], i, dir_path))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument(
        "--target",
        type=str,
        required=True,
    )
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    args = parser.parse_args()  # pylint: disable=invalid-name
    main()