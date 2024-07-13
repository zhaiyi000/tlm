# Enabling Tensor Language Model to Assist in Generating High-Performance Tensor Programs for Deep Learning

This repo is based on TVM [v0.12.0](https://github.com/apache/tvm/tree/v0.12.0) and reuses some code from [TenSet](https://github.com/tlc-pack/tenset) and [TLP](https://github.com/zhaiyi000/tlp).

TLM has been integrated into [Ansor](gen), [TVM(MetaSchedule)](meta), MindSpore's [AKG](https://github.com/mindspore-ai/akg) and AKG-MLIR.

[tlm slides](tlm%20slides.pptx)

## Installation

- Build and install this repo following the [guide](docs/install/from_source.rst).

- You can refer to my installation environment [here](version.log).

- [Note] To avoid this [issue](https://github.com/apache/tvm/issues/9362), please remember to set this👉. If you are a PyTorch user, it is recommended to set ``(USE_LLVM "/path/to/llvm-config --link-static")`` and ``set(HIDE_PRIVATE_SYMBOLS ON)`` to avoid potential symbol conflicts between different versions LLVM used by TVM and PyTorch.

- TLM uses huggingface for training and needs to install dependencies: 

  ```shell
  pip install -r requirements.txt
  ```

- You can download the [tlm_dataset](https://drive.google.com/file/d/1MdOxSIBFqYl1pWWUmj18vm4AG4Sbly8Z/view) we have collected and put them in the corresponding path.

  ```
  (py38) ➜  ~ tree tlm_dataset -L 2
  tlm_dataset
  ├── gen
  │   ├── dataset
  │   ├── gen_data
  │   └── utils.json
  └── meta
      ├── dataset
      ├── meta_data
      └── meta_utils.json
  
  7 directories, 2 files
  ```

## Getting Started Instructions

To get started quickly, you need to download [tlm_dataset](https://drive.google.com/file/d/1omj7AfPFIjuSgs-UBld0YRKWmmCe08G3/view?usp=drive_link). Here we take compiling `bert_base` as an example.

```shell
cd gen
```

- Optional. We have already trained TLM in `gen_data/v100_gen_best`. You can repeat the following command, which will overwrite `gen_data/v100_gen_best`. When executing `run_train_clm_best_v100.py`, you need to 1) adjust `CUDA_VISIBLE_DEVICES` and `--per_device_train_batch_size` and 2) `apt install tmux`.

  ```shell
  python postprocess.py --target=nvidia/nvidia-v100
  
  python make_dataset.py \
  --for_type=for_gen_best_all \
  --target=nvidia/nvidia-v100 \
  --dataset_path=dataset/measure_records/v100 \
  --tokenizer_path=gen_data/gen_tokenizer_v100 \
  --save_path=gen_data/v100_gen_best
  
  python run_train_clm_best_v100.py
  ```

- To generate tensor programs for bert_base, generate prompts first.

  ```shell
  python make_dataset.py \
  --for_type=for_gen_eval_sketch_only_bert \
  --target=nvidia/nvidia-v100 \
  --dataset_path=dataset/to_measure_programs/v100 \
  --tokenizer_path=gen_data/gen_tokenizer_v100 \
  --save_path=gen_data/v100_gen_eval_only_bert \
  --keep_cnt=64
  ```

- Generate tensor programs for bert_base.

  ```shell
  CUDA_VISIBLE_DEVICES=3 python gen_state.py \
  --model_name_or_path=gen_data/clm_gen_best_v100 \
  --sketch_path=gen_data/v100_gen_eval_only_bert/0_merge.json \
  --save_path=gen_data/v100_gen_eval_only_bert/gen_eval.json \
  --allow_repeat=True \
  --target=nvidia/nvidia-v100 \
  --keep_cnt=32
  ```

- Measure the execution latency of the generated tensor program. Ensure that the measurement hardware is exclusive to avoid inaccurate results.

  ```shell
  CUDA_VISIBLE_DEVICES=3 python measure_programs.py --batch-size=64 --target=nvidia/nvidia-v100 --to-measure-path=gen_data/v100_gen_eval_only_bert/gen_eval.json --measured-path=measured_only_bert.json
  ```

- Select the best-performing programs from the measured tensor programs to compile `bert_base` and measure the end-to-end execution latency of `bert_base`. Ensure that the measurement hardware is exclusive to avoid inaccurate results.

  ```shell
  CUDA_VISIBLE_DEVICES=3 TLM_LOG_FILE=measured_only_bert.json python tune_relay.py --workload=bert_base --input-shape=\[1,128\] --target=nvidia/nvidia-v100 --backend=graph
  ```

## TLM-Ansor

```shell
cd gen
```

1. Train TLM-base, taking the NVIDIA V100 as an example.

   - Partition the workload into subgraphs and save the results to the path `dataset/network_info/v100`.

     ```shell
     python dump_network_info.py --target=nvidia/nvidia-v100
     ```

     `--target`  can be found in `src/target/tag.cc` for other hardware. For CPUs, it can be set to something like this `--target="llvm -mcpu=core-avx2 -model=i7"`.

   - Dump tensor programs for those subgraphs. The resulting tensor programs have not measured execution latency and are unlabeled data. They will be saved to the path `dataset/to_measure_programs/v100`.

     ```shell
     python dump_programs.py --target=nvidia/nvidia-v100
     ```

   - Use unlabeled data to build a vocabulary and a tokenizer and save them to `--tokenizer_path`.

     ```shell
     python make_dataset.py \
     --for_type=for_gen_tokenizer \
     --target=nvidia/nvidia-v100 \
     --dataset_path=dataset/to_measure_programs/v100 \
     --tokenizer_path=gen_data/gen_tokenizer_v100
     ```

   - Use the tokenizer to train the TLM-base pre-train dataset and save it to `--save_path`.

     ```shell
     python make_dataset.py \
     --for_type=for_gen \
     --target=nvidia/nvidia-v100 \
     --dataset_path=dataset/to_measure_programs/v100 \
     --tokenizer_path=gen_data/gen_tokenizer_v100 \
     --save_path=gen_data/v100_gen_2154
     ```

   - Pre-train TLM-base. Adjust parameters such as `batch_size` in the run_train_clm.py file according to the GPU memory size. Requires `apt install tmux`.

     ```shell
     python run_train_clm.py
     ```

2. Train the TLM using iterative optimization. We provide two methods: the script with one kick and the step-by-step command.

   A. The script with one kick. The script uses a pipeline system and requires two machines. Both machines need to clone the TLM repository. The two machines communicate and exchange data using ssh and rsync.

   - On the training machine, 1) configure ssh password-free login, and then configure the target machine in `~/.ssh/config`, 2) set `device_id_all` in run.py to specify the GPU card IDs that can be used to train TLM; 3) Set `ssh_target` in run.py.

     ```shell
     python run.py --target=nvidia/nvidia-v100 --for_type=for_finetuning --finetuning_init=True
     ```

   - On the measurement machine, configure `available_ids` to specify the GPU card IDs that can be used for measurement.

     ```shell
     python run_measure.py
     ```

   B. Step-by-step command.

   - Before using TLM-base/TLM to generate tensor programs, generate prompts first.

     ```shell
     python make_dataset.py \
     --for_type=for_gen_train_sketch \
     --target=nvidia/nvidia-v100 \
     --dataset_path=dataset/to_measure_programs/v100 \
     --tokenizer_path=gen_data/gen_tokenizer_v100 \
     --save_path=gen_data/v100_gen_train \
     --keep_cnt=48 \
     --test_file_idx=0
     ```

   - Use TLM-base/TLM to generate tensor programs, `--model_name_or_path` specifies whether to use TLM-base or TLM.

     ```shell
     CUDA_VISIBLE_DEVICES=0,1,2,3 python gen_state.py \
     --target=nvidia/nvidia-v100 \
     --model_name_or_path=gen_data/clm_gen_v100/checkpoint-24000 \
     --sketch_path=gen_data/v100_gen_train/0_merge.json \
     --save_path=gen_data/v100_gen_train/gen_train.json \
     --allow_repeat=True \
     --keep_cnt=16
     ```

   - Measure the execution latency of the generated tensor program and 'manually' update the path of the measurement results to the `utils.json` file. There are many errors in the initial measurement data of iterative optimization. These errors are normal and will gradually decrease as the iteration proceeds.

     ```shell
     CUDA_VISIBLE_DEVICES=3 python measure_programs.py --batch-size=64 --target=nvidia/nvidia-v100 --to-measure-path=gen_data/v100_gen_train/gen_train.json --measured-path=gen_data/measure_data_v100/finetuning_0.json
     ```

   - Organize the measured programs into `dataset/measure_records/v100`.

     ```shell
     python postprocess.py --target=nvidia/nvidia-v100
     ```

   - Build an SFT dataset.

     ```shell
     python make_dataset.py \
     --for_type=for_gen_best \
     --target=nvidia/nvidia-v100 \
     --dataset_path=dataset/measure_records/v100 \
     --tokenizer_path=gen_data/gen_tokenizer_v100 \
     --save_path=gen_data/v100_gen_best
     ```

   - SFT TLM-base.

     ```shell
     python run_train_clm_best_v100.py
     ```

3. Evaluation on target workload.

   - Generate prompts.

     ```shell
     python make_dataset.py \
     --for_type=for_gen_eval_sketch \
     --target=nvidia/nvidia-v100 \
     --dataset_path=dataset/to_measure_programs/v100 \
     --tokenizer_path=gen_data/gen_tokenizer_v100 \
     --save_path=gen_data/v100_gen_eval \
     --keep_cnt=64
     ```

   - Generate tensor programs.

     ```shell
     CUDA_VISIBLE_DEVICES=4 python gen_state.py \
     --model_name_or_path=gen_data/clm_gen_best_v100 \
     --sketch_path=gen_data/v100_gen_eval/0_merge.json \
     --save_path=gen_data/v100_gen_eval/gen_eval.json \
     --allow_repeat=True \
     --target=nvidia/nvidia-v100 \
     --keep_cnt=32
     ```

   - Measure the execution latency of the generated tensor program.

     ```shell
     CUDA_VISIBLE_DEVICES=3 python measure_programs.py --batch-size=64 --target=nvidia/nvidia-v100 --to-measure-path=gen_data/v100_gen_eval/gen_eval.json --measured-path=gen_data/measure_data_v100/0_test_3.json
     ```

   - Use scripts to analyze the speedups.

     ```shell
     python speedup_eval.py --target=nvidia/nvidia-v100 --for_test=True
     ```

4. When the tuning budget is ample, we continue to optimize TLM using the target workload data. There are also two ways.

   A. The script with one kick.

   - On the training machine.

     ```shell
     python run.py --target=nvidia/nvidia-v100 --for_type=for_testtuning --testtuning_init=True
     ```

   - On the measurement machine.

     ```shell
     python run_measure.py
     ```

   B. Step-by-step command.

   - Generate prompts.

     ```shell
     python make_dataset.py \
     --for_type=for_gen_evaltuning_sketch \
     --target=nvidia/nvidia-v100 \
     --dataset_path=dataset/to_measure_programs/v100 \
     --tokenizer_path=gen_data/gen_tokenizer_v100 \
     --save_path=gen_data/v100_gen_evaltuning \
     --keep_cnt=64
     ```

   - Generate tensor programs.

     ```shell
     CUDA_VISIBLE_DEVICES=0,1,2,3 python gen_state.py \
     --model_name_or_path=gen_data/clm_gen_best_v100 \
     --sketch_path=gen_data/v100_gen_evaltuning/0_merge.json \
     --save_path=gen_data/v100_gen_evaltuning/gen_eval.json \
     --allow_repeat=True \
     --target=nvidia/nvidia-v100 \
     --keep_cnt=32
     ```

   - Measure the execution latency of the generated tensor program and 'manually' update the path of the measurement results to the `utils.json` file.

     ```shell
     CUDA_VISIBLE_DEVICES=3 python measure_programs.py --batch-size=64 --target=nvidia/nvidia-v100 --to-measure-path=gen_data/v100_gen_evaltuning/gen_eval.json --measured-path=gen_data/measure_data_v100/testtuning_0.json
     ```

   - Organize the measured programs into `dataset/measure_records/v100`.

     ```shell
     python postprocess.py --target=nvidia/nvidia-v100
     ```

   - Build an SFT dataset.

     ```shell
     python make_dataset.py \
     --for_type=for_gen_best_all \
     --target=nvidia/nvidia-v100 \
     --dataset_path=dataset/measure_records/v100 \
     --tokenizer_path=gen_data/gen_tokenizer_v100 \
     --save_path=gen_data/v100_gen_best
     ```

   - SFT TLM-base

     ```shell
     python run_train_clm_best_v100.py
     ```

   - Not every task has the same optimization space. We use the task scheduler to allocate the tuning budget.

     ```shell
     python task_sheduler.py --target=nvidia/nvidia-v100 --for_testtuning=True
     ```

## TLM-Meta

```shell
cd meta
```

Similar to TLM-Ansor, command lines can be found in run.sh and run.py.

## License

TLM is licensed under the [Apache-2.0](https://github.com/apache/tvm/blob/main/LICENSE) license.