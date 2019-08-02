#!/bin/bash

#SBATCH -J TENT                    # Job name
#SBATCH -o TENT.out             # Name of stdout output file (%j expands to %jobID)
#SBATCH -t 700:00:00              # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -N 1 
#SBATCH --array=0-25

module load py36-brian2/2.2.2.1
python3 tent_map.py $SLURM_ARRAY_TASK_ID
