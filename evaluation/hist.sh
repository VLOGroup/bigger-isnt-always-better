#!/bin/bash
#SBATCH --partition=icg
#SBATCH --nodes=1
#SBATCH --nodelist=nvcluster-node4

source activate diff_mri
cd /home/lg/score-MRI/diffusion-generalization-mri

python hist.py --model fastmri_knee_1