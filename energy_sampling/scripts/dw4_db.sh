for seed in 0
do
    CUDA_VISIBLE_DEVICES=0, python train.py \
        --seed $seed \
        --t_scale 1. --energy dw4 --pis_architectures --zero_init --clipping\
        --mode_fwd db --lr_policy 1e-4 --lr_flow 1e-3 \
        --conditional_flow_model \
        --exploratory --exploration_wd --exploration_factor 0.2 \
        --hidden_dim 256 --t_emb_dim 256 --batch_size 256 --epochs 50000
done