#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ivc
#SBATCH --ntasks=1

source activate diff_mri
cd /home/lg/ijcv/ijcv

for algo in dps pc ald
do
    for set in corpd corpdfs brain
    do
        srun python evaluation_main.py \
            --model_config=configs/models/mrncsn_structured_celeba.yaml \
            --data_config=configs/datasets/fast_mri_${set}.yaml \
            --evaluation_config=configs/evaluation/gaussian_1d_8.yaml \
            --sample_config=configs/samplers/${algo}_sampler.yaml \
            --savedir=/srv/local/lg/results/mrncsn_structured_celeba_${set}_g1d8_${algo}
    done
done
