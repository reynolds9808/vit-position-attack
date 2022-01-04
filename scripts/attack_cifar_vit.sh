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
    --dataset_name cifar100\
    --model_name_or_path  ../model/1e-4_cifar100_vit_finelabel/\
    --attack_model_dir ../model/attack_cifar100_vit/ \
    --output_dir ../model/1e-4_cifar100_vit_finelabel/ \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --test_type random \
    --learning_rate 1e-2 \
    --num_labels 100 \
    --output_logs_path ../log/adv_cifar100_2_FixedPGD.logs \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --attack_flag FixedPGD \
    --top_k 2\
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337