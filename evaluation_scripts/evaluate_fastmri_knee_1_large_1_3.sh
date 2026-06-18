#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=l40

source activate ijcv

for algo in dps ald pc
do
    for set in brain
    do
        for model in 1_large
        do
            for mask in gaussian_1d_4 gaussian_1d_8 gaussian_2d_4 radial poisson
            do
                # srun python evaluation_main.py \
                #     --model_config=configs/models/ncsnpp_fastmr_${model}.yaml \
                #     --data_config=configs/datasets/fast_mri_${set}.yaml \
                #     --evaluation_config=configs/evaluation/${mask}.yaml \
                #     --sample_config=configs/samplers/${algo}_sampler.yaml \
                #     --savedir=/srv/local/lg/ijcv_update/fastmri_${model}_${set}_${mask}_${algo}

                srun python evaluation_main.py \
                    --model_config=configs/models/ncsnpp_celeba_${model}.yaml \
                    --data_config=configs/datasets/fast_mri_${set}.yaml \
                    --evaluation_config=configs/evaluation/${mask}.yaml \
                    --sample_config=configs/samplers/${algo}_sampler.yaml \
                    --savedir=/srv/local/lg/ijcv_update/celeba_${model}_${set}_${mask}_${algo}
            done
        done
    done
done
