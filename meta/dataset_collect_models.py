# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring

import argparse
import os
from typing import List, Tuple

from tqdm import tqdm  # type: ignore
import tvm
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.relay_integration import extracted_tasks_to_tune_contexts, extract_tasks
import pickle
from meta_common import get_task_info_filename, register_data_path
import glob
import tempfile
from tvm import meta_schedule as ms


# pylint: disable=too-many-branches
def _build_dataset() -> List[Tuple[str, List[int]]]:
    network_keys = []
    for name in [
        "resnet_18",
        "resnet_50",
        "mobilenet_v2",
        "mobilenet_v3",
        "wide_resnet_50",
        "resnext_50",
        "densenet_121",
        "vgg_16",
    ]:
        for batch_size in [1, 4, 8]:
            for image_size in [224, 240, 256]:
                network_keys.append((name, [batch_size, 3, image_size, image_size]))
    # inception-v3
    for name in ["inception_v3"]:
        for batch_size in [1, 2, 4]:
            for image_size in [299]:
                network_keys.append((name, [batch_size, 3, image_size, image_size]))
    # resnet3d
    for name in ["resnet3d_18"]:
        for batch_size in [1, 2, 4]:
            for image_size in [112, 128, 144]:
                network_keys.append((name, [batch_size, 3, image_size, image_size, 16]))
    # bert
    for name in ["bert_tiny", "bert_base", "bert_medium", "bert_large"]:
        for batch_size in [1, 2, 4]:
            for seq_length in [64, 128, 256]:
                network_keys.append((name, [batch_size, seq_length]))
    # dcgan
    for name in ["dcgan"]:
        for batch_size in [1, 4, 8]:
            for image_size in [64]:
                network_keys.append((name, [batch_size, 3, image_size, image_size]))
    return network_keys


def build_network_keys():
    keys = _build_dataset()
    return keys


def dump_network(keys, target):
    for name, input_shape in tqdm(keys):
        task_info_filename = get_task_info_filename((name, input_shape), target)
        if os.path.exists(task_info_filename):
            continue
        mod, params, inputs = get_network(name=name, input_shape=input_shape)
        extracted_tasks=extract_tasks(
            mod,
            target,
            params,
            module_equality='structural',
            disabled_pass=None,
            instruments=None,
        )
        with open(task_info_filename, 'wb') as f:
            pickle.dump(extracted_tasks, f)


def get_all_tasks():
    work_dir = tempfile.TemporaryDirectory()
    database = ms.database.JSONDatabase(work_dir=work_dir.name, module_equality='structural')
    from meta_common import NETWORK_INFO_FOLDER
    files = glob.glob(f'{NETWORK_INFO_FOLDER}/*.task.pkl')
    extracted_tasks = []
    hash_map = {}
    for file in tqdm(files):
        with open(file, 'rb') as f:
            tasks = pickle.load(f)
        filename = os.path.splitext(file)[0]
        for task_i, task in enumerate(tasks):
            hash = database.get_hash(task.dispatched[0])
            if hash in hash_map:
                if database.check_equal(hash_map[hash], task.dispatched[0]):
                    print('duplication')
                    continue
                else:
                    assert(False)
            hash_map[hash] = task.dispatched[0]
            extracted_tasks.append(task)
        with open(file, 'wb') as f:
            pickle.dump(tasks, f)
    work_dir.cleanup()
    # extracted_tasks.sort(key=lambda x: x[0])
    return extracted_tasks


def main():
    keys = _build_dataset()
    register_data_path(args.target)
    from meta_common import NETWORK_INFO_FOLDER
    os.makedirs(NETWORK_INFO_FOLDER, exist_ok=True)
    target = tvm.target.Target(args.target)
    dump_network(keys, target)

    tasks = get_all_tasks()
    print(len(tasks))
    pickle.dump(tasks, open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args = parser.parse_args()  # pylint: disable=invalid-name
    main()
