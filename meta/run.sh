

python dataset_collect_models.py --target=nvidia/nvidia-v100
# Note that when the target is CPU, you need to specify '-num-cores'
# such as   '--target="llvm -mcpu=core-avx2 -model=i7 -num-cores=4"'
# In ansor, '--target="llvm -mcpu=core-avx2 -model=i7"' is ok.


python dump_programs.py --target=nvidia/nvidia-v100




python meta_make_dataset.py \
--for_type=for_gen_tokenizer \
--target=nvidia/nvidia-v100 \
--tokenizer_path=meta_data/v100_tokenizer \
--dataset_path=dataset/to_measure_programs/v100


python meta_make_dataset.py \
--for_type=for_gen \
--target=nvidia/nvidia-v100 \
--tokenizer_path=meta_data/v100_tokenizer \
--save_path=meta_data/v100_gen \
--dataset_path=dataset/to_measure_programs/v100

python run_train_clm_v100.py


python run.py --target=nvidia/nvidia-v100 --for_type=for_finetuning --finetuning_init=True
python run.py --target=nvidia/nvidia-v100 --for_type=for_testtuning --testtuning_init=True

