#!/bin/sh
# SLURM SCRIPTS

#SBATCH --nodes=1
#SBATCH -p dell
#SBATCH -c 32
#SBATCH --gres=gpu:V100:1
#SBATCH -x dell-gpu-24
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -o ../log/test_cifar10.logs


# -------------------------
# debugging flags (optional)
#  export NCCL_DEBUG=INFO
#  export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest cuda
# module load NCCL/2.4.7-1-cuda.10.0
# -------------------------
python ../src/adv_image_classification.py \
    --dataset_name cifar10\
    --model_name_or_path  ../model/1e-4_cifar10_vit/\
    --attack_model_dir ../model/attack_cifar10_vit/ \
    --output_dir ../model/1e-4_cifar10_vit/ \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --test_type random \
    --learning_rate 1e-1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337