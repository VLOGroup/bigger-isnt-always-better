#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ivc
#SBATCH --ntasks=1
#SBATCH --nodelist=nvcluster-node4
#SBATCH --mail-user=lukas.glaszner@tugraz.at
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source activate diff_mri
cd /home/lg/ijcv/ijcv

for algo in dps
do
    for set in corpd corpdfs brain
    do
        for model in 4_attention 4 3 2 1
        srun python evaluation_main.py \
            --model_config=configs/models/ncsnpp_fastmri_${model}.yaml \
            --data_config=configs/datasets/fast_mri_${set}.yaml \
            --evaluation_config=configs/evaluation/poisson.yaml \
            --sample_config=configs/samplers/${algo}_sampler.yaml \
            --savedir=/srv/local/lg/results/fastmri_${model}_${set}_poisson_${algo}

        srun python evaluation_main.py \
            --model_config=configs/models/ncsnpp_celeba_${model}.yaml \
            --data_config=configs/datasets/fast_mri_${set}.yaml \
            --evaluation_config=configs/evaluation/poisson.yaml \
            --sample_config=configs/samplers/${algo}_sampler.yaml \
            --savedir=/srv/local/lg/results/celeba_${model}_${set}_poisson_${algo}
    done
done
