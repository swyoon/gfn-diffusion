for seed in 0 1 2
do
    CUDA_VISIBLE_DEVICES=1, python train.py \
        --seed $seed \
        --t_scale 3.0 \
        --energy dw4 \
        --pis_architectures \
        --zero_init \
        --clipping \
        --mode_fwd tb \
        --lr_policy 1e-4 \
        --lr_flow 1e-2\
        --exploratory --exploration_wd --exploration_factor 0.2 \
        --hidden_dim 256 --t_emb_dim 256 --batch_size 256

done