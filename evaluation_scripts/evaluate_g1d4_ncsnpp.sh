export CUDA_VISIBLE_DEVICES=0

source activate diff_mri
cd ..

for algo in dps
do
    for set in corpdfs # corpdfs brain
    do
        for model in 4_attention 4 3 2 1
        do
            python evaluation_main.py \
                --model_config=configs/models/ncsnpp_fastmri_${model}.yaml \
                --data_config=configs/datasets/fast_mri_${set}.yaml \
                --evaluation_config=configs/evaluation/radial.yaml \
                --sample_config=configs/samplers/${algo}_sampler.yaml \
                --savedir=/media/lukasglaszner/data/results/fastmri_${model}_${set}_radial_${algo}_no_speedup

            # python evaluation_main.py \
            #     --model_config=configs/models/ncsnpp_celeba_${model}.yaml \
            #     --data_config=configs/datasets/fast_mri_${set}.yaml \
            #     --evaluation_config=configs/evaluation/gaussian_1d_4.yaml \
            #     --sample_config=configs/samplers/${algo}_sampler.yaml \
            #     --savedir=/media/lukasglaszner/data/results/celeba_${model}_${set}_g1d4_${algo}
        done
    done
done
