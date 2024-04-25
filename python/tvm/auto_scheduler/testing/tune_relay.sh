set -x

# export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1


python tune_relay.py --workload=resnet_50 --input-shape=[1,3,224,224] --target='nvidia/nvidia-v100' --num-trials=20000 --work-dir=./records --backend=graph
python tune_relay.py --workload=mobilenet_v2 --input-shape=[1,3,224,224] --target='nvidia/nvidia-v100' --num-trials=20000 --work-dir=./records --backend=graph
python tune_relay.py --workload=resnext_50 --input-shape=[1,3,224,224] --target='nvidia/nvidia-v100' --num-trials=20000 --work-dir=./records --backend=graph
python tune_relay.py --workload=bert_base --input-shape=[1,128] --target='nvidia/nvidia-v100' --num-trials=20000 --work-dir=./records --backend=graph
python tune_relay.py --workload=bert_tiny --input-shape=[1,128] --target='nvidia/nvidia-v100' --num-trials=20000 --work-dir=./records --backend=graph

python tune_relay.py --workload=densenet_121 --input-shape=[8,3,256,256] --target='nvidia/nvidia-v100' --num-trials=20000 --work-dir=./records --backend=graph
python tune_relay.py --workload=bert_large --input-shape=[4,256] --target='nvidia/nvidia-v100' --num-trials=20000 --work-dir=./records --backend=graph
python tune_relay.py --workload=wide_resnet_50 --input-shape=[8,3,256,256] --target='nvidia/nvidia-v100' --num-trials=20000 --work-dir=./records --backend=graph
python tune_relay.py --workload=resnet3d_18 --input-shape=[4,3,144,144,16] --target='nvidia/nvidia-v100' --num-trials=20000 --work-dir=./records --backend=graph
python tune_relay.py --workload=dcgan --input-shape=[8,3,64,64] --target='nvidia/nvidia-v100' --num-trials=20000 --work-dir=./records --backend=graph



