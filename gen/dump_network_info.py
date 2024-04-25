import gc
import glob
import os
import pickle
import argparse
from tqdm import tqdm
import tvm
from tvm import relay
from tvm import auto_scheduler
from common import get_relay_ir_filename, get_task_info_filename, register_data_path
from tvm.meta_schedule.testing.dataset_collect_models import build_network_keys
from tvm.meta_schedule.testing.relay_workload import get_network


def dump_network(network_key, target, hardware_params):
    name, args = network_key
    network_task_key = (network_key,) + (target,)

    relay_ir_filename = get_relay_ir_filename(target, network_key)
    task_info_filename = get_task_info_filename(network_key, target)

    if os.path.exists(task_info_filename):
        return

    mod, params, inputs = get_network(*network_key)

    # Dump network relay ir
    if not os.path.exists(relay_ir_filename):
        print(f"Dump relay ir for {network_key}...")
        mod_json = tvm.ir.save_json(mod)
        params_bytes = relay.save_param_dict(params)
        pickle.dump((mod_json, len(params_bytes), inputs),
                    open(relay_ir_filename, "wb"))

    # Dump task information
    if not os.path.exists(task_info_filename):
        print(f"Dump task info for {network_task_key}...")
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, tvm.target.Target(target), hardware_params=hardware_params)
        pickle.dump((tasks, task_weights), open(task_info_filename, "wb"))


def get_all_tasks():
    all_task_keys = set()
    all_tasks = []
    duplication = 0

    filenames = glob.glob(f"{NETWORK_INFO_FOLDER}/*.task.pkl")
    filenames.sort()

    for filename in tqdm(filenames):
        tasks, task_weights = pickle.load(open(filename, "rb"))
        for t in tasks:
            task_key = (t.workload_key, str(t.target.kind))

            if task_key not in all_task_keys:
                all_task_keys.add(task_key)
                all_tasks.append(t)
            else:
                duplication += 1

    return all_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    register_data_path(args.target)
    args.target = tvm.target.Target(args.target)

    from common import NETWORK_INFO_FOLDER
    assert(NETWORK_INFO_FOLDER is not None)
    os.makedirs(NETWORK_INFO_FOLDER, exist_ok=True)

    # Dump the relay ir and task info for all networks
    network_keys = build_network_keys()

    if args.target.kind.name == "llvm":
        hardware_params = auto_scheduler.HardwareParams(target=args.target)
    elif args.target.kind.name == "cuda":
        hardware_params = auto_scheduler.HardwareParams(
            num_cores=-1,
            vector_unit_bytes=16,
            cache_line_bytes=64,
            max_shared_memory_per_block=int(args.target.attrs["max_shared_memory_per_block"]),
            max_threads_per_block=int(args.target.attrs["max_threads_per_block"]),
            # The value `max_local_memory_per_block` is not used in AutoScheduler,
            # but is required by the API.
            max_local_memory_per_block=12345678,
            max_vthread_extent=8,
            warp_size=32,
        )
    else:
        raise NotImplementedError(f"Unsupported target {args.target}")
    for key in tqdm(network_keys):
        dump_network(key, args.target, hardware_params)
        gc.collect()

    # Dump an index table that contains all tasks
    tasks = get_all_tasks()
    tasks.sort(key=lambda x: (str(x.target.kind), x.compute_dag.flop_ct, x.workload_key))
    pickle.dump(tasks, open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "wb"))
