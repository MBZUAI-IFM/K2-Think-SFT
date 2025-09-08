#!/bin/bash
#SBATCH --job-name=qwen-32b-am-3697c18
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8

srun --cpu-bind=none bash -c 'export HF_HOME=/path/hf_cache_$(hostname)_$SLURM_PROCID && mkdir -p $HF_HOME && mkdir -p $HF_HOME'

source /path/conda.sh
conda activate env

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
echo "Nodes to check: ${nodes[@]}"

echo "SLURM_NNODES: $SLURM_NNODES"
echo $HF_HOME
total_gpus=$((SLURM_NNODES * 8))
echo "total_gpus: $total_gpus"
host_list=""
for node in "${nodes[@]}"; do
    if [ -z "$host_list" ]; then
        host_list="$node"
    else
        host_list="$host_list,$node"
    fi
done

export WANDB_API_KEY=wandb_api_key
# Get master node information
MASTER_NODE_IP=$(scontrol show hostnames | head -n 1)

# Launch distributed training using srun
srun --cpu-bind=none \
    bash -c 'HF_HOME=/path/hf_cache_$(hostname)_$SLURM_PROCID \
    TRITON_HOME=/tmp/triton_cache \
    FORCE_TORCHRUN=1 NNODES='$SLURM_NNODES' NODE_RANK=$SLURM_PROCID MASTER_ADDR='$MASTER_NODE_IP' MASTER_PORT=29500 \
    llamafactory-cli train examples/train_full/Qwen2.5-32B-base-AM-Thinking-v1-Distilled-3697c18.yaml'
