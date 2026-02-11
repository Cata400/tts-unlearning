#!/bin/bash
#SBATCH --job-name=catalin_unlearn
#SBATCH --time=99:00:00  # hh:mm:ss . It is usually a good idea to limit your job.
#SBATCH --output=./remote/logs/output_%A.log   # %x_%j_%N.log # %A is the job id
#SBATCH --error=./remote/logs/error_%A.log  # %x_%j_%N.log # %A is the job id
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p supermicro


########### Arguments ###########
### $1: config
### $2: number of gpus (optional, default: 1)
#################################

CONFIG=$1
echo "CONFIG: $CONFIG"

# if the second argument exists, use it, otherwise use the default value
if [ -z "$2" ]
then
    echo "Using default N_GPUS: 1"
    N_GPUS=1
else
    echo "N_GPUS: $2"
    N_GPUS=$2
fi


source /etc/profile.d/modules.sh
module load anaconda/3
module load cuda/12.4-9.1
conda activate f5-tts

if N_GPUS -gt 1
then
    accelerate launch ../train/unlearn.py --config $CONFIG
else
    python3 ../train/unlearn.py $CONFIG
