export CUDA_VISIBLE_DEVICES=2

source activate diff_mri
cd /home/glaszner/ijcv/ijcv

for algo in dps pc ald
do
    for set in thorax head
    do
        python -u evaluation_main.py \
            --model_config=configs/models/mrncsn_structured_celeba.yaml \
            --data_config=configs/datasets/ct_${set}.yaml \
            --evaluation_config=configs/evaluation/limited_angle_90.yaml \
            --sample_config=configs/samplers/${algo}_sampler.yaml \
            --savedir=/mount/data/glaszner/results/mrncsn_structured_celeba_${set}_la90_${algo}
    done
done
