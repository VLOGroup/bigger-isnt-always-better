export CUDA_VISIBLE_DEVICES=2

source activate diff_mri

for algo in dps ald pc
do
    for set in thorax head
    do
        for model in 4_small
        do
            for mask in fanbeam_30 sparse_view_60 sparse_view_30 sparse_view_20
            do
                python evaluation_main.py \
                    --model_config=configs/models/ncsnpp_fastmri_${model}.yaml \
                    --data_config=configs/datasets/ct_${set}.yaml \
                    --evaluation_config=configs/evaluation/${mask}.yaml \
                    --sample_config=configs/samplers/${algo}_sampler.yaml \
                    --savedir=/mount/data/glaszner/ijcv_update/ct_${model}_${set}_${mask}_${algo}

                python evaluation_main.py \
                    --model_config=configs/models/ncsnpp_celeba_${model}.yaml \
                    --data_config=configs/datasets/ct_${set}.yaml \
                    --evaluation_config=configs/evaluation/${mask}.yaml \
                    --sample_config=configs/samplers/${algo}_sampler.yaml \
                    --savedir=/mount/data/glaszner/ijcv_update/celeba_${model}_${set}_${mask}_${algo}
            done
        done
    done
done
