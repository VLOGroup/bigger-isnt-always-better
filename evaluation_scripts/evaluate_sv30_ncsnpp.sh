#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=ivc
#SBATCH --ntasks=1
#SBATCH --nodelist=nvcluster-node4
#SBATCH --mail-user=lukas.glaszner@tugraz.at
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN

source activate diff_mri
cd /home/lg/ijcv/ijcv

for algo in dps ald pc
do
    for set in thorax brain
    do
        for model in 4_attention 4 3 2 1
        do
            srun python evaluation_main.py \
                --model_config=configs/models/ncsnpp_ct_${model}.yaml \
                --data_config=configs/datasets/ct_${set}.yaml \
                --evaluation_config=configs/evaluation/sparse_view_30.yaml \
                --sample_config=configs/samplers/${algo}_sampler_ct.yaml \
                --savedir=/srv/local/lg/results/ct_${model}_${set}_sv30_${algo}

            srun python evaluation_main.py \
                --model_config=configs/models/ncsnpp_celeba_${model}.yaml \
                --data_config=configs/datasets/ct_${set}.yaml \
                --evaluation_config=configs/evaluation/sparse_view_30.yaml \
                --sample_config=configs/samplers/${algo}_sampler_ct.yaml \
                --savedir=/srv/local/lg/results/celeba_${model}_${set}_sv30_${algo}
        done
    done
done
