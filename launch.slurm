#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --job-name=tap
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --output=slurm_jobs/myjob.%j.out
#SBATCH --error=slurm_jobs/myjob.%j.err

knocky python automate_all.py