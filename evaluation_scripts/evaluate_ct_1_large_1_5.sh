#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=l40

source activate ijcv

for algo in ald
do
    for set in head
    do
        for model in 1_large
        do
            for mask in fanbeam_60
            do
                srun python evaluation_main.py \
                    --model_config=configs/models/ncsnpp_ct_${model}.yaml \
                    --data_config=configs/datasets/ct_${set}.yaml \
                    --evaluation_config=configs/evaluation/${mask}.yaml \
                    --sample_config=configs/samplers/${algo}_sampler_ct.yaml \
                    --savedir=/srv/local/lg/ijcv_update/ct_${model}_${set}_${mask}_${algo}

                # srun python evaluation_main.py \
                #     --model_config=configs/models/ncsnpp_celeba_${model}.yaml \
                #     --data_config=configs/datasets/ct_${set}.yaml \
                #     --evaluation_config=configs/evaluation/${mask}.yaml \
                #     --sample_config=configs/samplers/${algo}_sampler_ct.yaml \
                #     --savedir=/srv/local/lg/ijcv_update/celeba_${model}_${set}_${mask}_${algo}
            done
        done
    done
done
