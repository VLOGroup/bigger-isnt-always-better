#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=l40
#SBATCH --mail-user=lukas.glaszner@tugraz.at
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN

source activate diff_mri

for algo in ald
do
    for set in head thorax
    do
        for model in 4_attention 4 4_small
        do
            python evaluation_main.py \
                --model_config=configs/models/ncsnpp_ct_${model}.yaml \
                --data_config=configs/datasets/ct_${set}.yaml \
                --evaluation_config=configs/evaluation/fanbeam_60.yaml \
                --sample_config=configs/samplers/${algo}_sampler_ct.yaml \
                --savedir=/srv/local/lg/ijcv_update_fb/ct_${model}_${set}_fanbeam_60_${algo}

            python evaluation_main.py \
                --model_config=configs/models/ncsnpp_celeba_${model}.yaml \
                --data_config=configs/datasets/ct_${set}.yaml \
                --evaluation_config=configs/evaluation/fanbeam_60.yaml \
                --sample_config=configs/samplers/${algo}_sampler_ct.yaml \
                --savedir=/srv/local/lg/ijcv_update_fb/celeba_${model}_${set}_fanbeam_60_${algo}
        done
    done
done
