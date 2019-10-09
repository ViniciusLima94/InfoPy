#!/bin/bash

#SBATCH -J ULAM                    # Job name
#SBATCH -o out/ULAM.out             # Name of stdout output file (%j expands to %jobID)
#SBATCH -t 700:00:00              # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -N 1 
#SBATCH --array=1-51

module load py36-brian2/2.2.2.1
python3 ulam_map.py $SLURM_ARRAY_TASK_ID
