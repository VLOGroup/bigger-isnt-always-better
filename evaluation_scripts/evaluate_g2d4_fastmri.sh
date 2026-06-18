export CUDA_VISIBLE_DEVICES=0

source activate diff_mri
cd ..

for algo in dps pc ald
do
    for set in corpd corpdfs brain
    do
        python evaluation_main.py \
            --model_config=configs/models/mrncsn_structured.yaml \
            --data_config=configs/datasets/fast_mri_${set}.yaml \
            --evaluation_config=configs/evaluation/gaussian_2d_4.yaml \
            --sample_config=configs/samplers/${algo}_sampler.yaml \
            --savedir=/media/lukasglaszner/data/results/mrncsn_structured_fastmri_${set}_g2d4_${algo}
    done
done

for algo in dps pc ald
do
    for set in corpd corpdfs brain
    do
        python evaluation_main.py \
            --model_config=configs/models/mrncsn_structured.yaml \
            --data_config=configs/datasets/fast_mri_${set}.yaml \
            --evaluation_config=configs/evaluation/gaussian_1d_4.yaml \
            --sample_config=configs/samplers/${algo}_sampler.yaml \
            --savedir=/media/lukasglaszner/data/results/mrncsn_structured_fastmri_${set}_g1d4_${algo}
    done
done
