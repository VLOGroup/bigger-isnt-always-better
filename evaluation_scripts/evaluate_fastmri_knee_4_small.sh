export CUDA_VISIBLE_DEVICES=3

source activate diff_mri

for algo in dps ald pc
do
    for set in corpd corpdfs brain
    do
        for model in 4_small
        do
            for mask in gaussian_1d_4 gaussian_1d_8 gaussian_2d_4 radial poisson
            do
                python evaluation_main.py \
                    --model_config=configs/models/ncsnpp_fastmri_${model}.yaml \
                    --data_config=configs/datasets/fast_mri_${set}.yaml \
                    --evaluation_config=configs/evaluation/${mask}.yaml \
                    --sample_config=configs/samplers/${algo}_sampler.yaml \
                    --savedir=/mount/data/glaszner/ijcv_update/fastmri_${model}_${set}_${mask}_${algo}

                python evaluation_main.py \
                    --model_config=configs/models/ncsnpp_celeba_${model}.yaml \
                    --data_config=configs/datasets/fast_mri_${set}.yaml \
                    --evaluation_config=configs/evaluation/${mask}.yaml \
                    --sample_config=configs/samplers/${algo}_sampler.yaml \
                    --savedir=/mount/data/glaszner/ijcv_update/celeba_${model}_${set}_${mask}_${algo}
            done
        done
    done
done
