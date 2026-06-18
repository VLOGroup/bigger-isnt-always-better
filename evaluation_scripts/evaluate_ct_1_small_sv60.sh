#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=rtx8000
#SBATCH --ntasks=1
#SBATCH --nodelist=nvcluster-node4
#SBATCH --mail-user=lukas.glaszner@tugraz.at
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source activate diff_mri
cd /home/lg/ijcv/ijcv

for algo in dps ald pc
do
    for set in head thorax
    do
        for model in 1_small
        do
            srun python evaluation_main.py \
                --model_config=configs/models/ncsnpp_ct_${model}.yaml \
                --data_config=configs/datasets/ct_${set}.yaml \
                --evaluation_config=configs/evaluation/sparse_view_60.yaml \
                --sample_config=configs/samplers/${algo}_sampler.yaml \
                --savedir=/srv/local/lg/ijcv_update/ct_${model}_${set}_sv60_${algo}

            srun python evaluation_main.py \
                --model_config=configs/models/ncsnpp_celeba_${model}.yaml \
                --data_config=configs/datasets/ct_${set}.yaml \
                --evaluation_config=configs/evaluation/sparse_view_60.yaml \
                --sample_config=configs/samplers/${algo}_sampler.yaml \
                --savedir=/srv/local/lg/ijcv_update/celeba_${model}_${set}_sv60_${algo}
        done
    done
done
