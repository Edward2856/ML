#!/bin/bash
#SBATCH --job-name=BNN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=BNN_%j.out

python BNN_pytorch.py

python Real_weighted_model_pytorch.py