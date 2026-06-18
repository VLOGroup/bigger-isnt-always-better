#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=rtx8000
#SBATCH --nodelist=nvcluster-node3
#SBATCH --mail-user=lukas.glaszner@tugraz.at
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN

source activate diff_mri

for algo in dps ald pc
do
    for set in head
    do
        for model in 4_small 4 1
        do
            python evaluation_main.py \
                --model_config=configs/models/ncsnpp_ct_${model}.yaml \
                --data_config=configs/datasets/ct_${set}.yaml \
                --evaluation_config=configs/evaluation/fanbeam_45.yaml \
                --sample_config=configs/samplers/${algo}_sampler_ct.yaml \
                --savedir=/srv/local/lg/ijcv_update_test_comp/ct_${model}_${set}_fanbeam_45_${algo}_comp

            # python evaluation_main.py \
            #     --model_config=configs/models/ncsnpp_celeba_${model}.yaml \
            #     --data_config=configs/datasets/ct_${set}.yaml \
            #     --evaluation_config=configs/evaluation/fanbeam_60.yaml \
            #     --sample_config=configs/samplers/${algo}_sampler_ct.yaml \
            #     --savedir=/srv/local/lg/ijcv_update_test/celeba_${model}_${set}_fanbeam_60_${algo}_la
        done
    done
done
