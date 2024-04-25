import subprocess, os

# 设置 screen 命令和相关参数
session_name = os.path.basename(os.path.abspath(__file__))
log_file = f'{session_name}.log'
session_name = session_name.replace('.', '_')

if os.path.exists(log_file):
    tag = input(log_file + ' exist, delete it? [n]')
    if tag == 'y':
        # 删除文件
        os.remove(log_file)

# 构建完整的 screen 命令
cmd = """tmux new -s %s -d '{ 
{ 
set -x
echo "#################################################################"
date

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_clm.py \
                                    --do_train \
                                    --model_type=gpt2 \
                                    --tokenizer_name=gen_data/gen_tokenizer_v100 \
                                    --output_dir=gen_data/clm_gen_v100 \
                                    --dataset_name=gen_data/v100_gen_2154 \
                                    --per_device_train_batch_size=5 \
                                    \
                                    --overwrite_output_dir=True \
                                    --logging_steps=100 \
                                    --num_train_epochs=3 \
                                    --remove_unused_columns=False \
                                    --learning_rate=5e-5 \
                                    --save_steps=4000 \
                                    # --model_name_or_path=clm_gen_v100/checkpoint-56000
                                    # --resume_from_checkpoint=clm_gen/checkpoint-52000


                                    
                                    # --do_eval \
                                    # --per_device_eval_batch_size=128 \
                                    # --evaluation_strategy=steps \
                                    # --eval_steps=80000000000000 \


date
} |& tee -a %s 
}' 
""" % (session_name, log_file)

# 使用 subprocess 运行命令
subprocess.Popen(cmd, shell=True)