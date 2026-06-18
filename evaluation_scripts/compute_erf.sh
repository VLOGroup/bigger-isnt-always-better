export CUDA_VISIBLE_DEVICES=0

source activate diff_mri

python -u compute_erf.py \
    --model_config=configs/models/ncsnpp_fastmri_2.yaml \
    --num_iter=1000 \
    --savedir=/media/lukasglaszner/data/ijcv_update/ncsnpp_fastmri_2_erf