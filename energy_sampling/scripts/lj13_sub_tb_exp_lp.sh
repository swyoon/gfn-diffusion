for seed in 0
do
    CUDA_VISIBLE_DEVICES=0, python train.py \
        --seed $seed \
        --t_scale 1. --energy lj13 --pis_architectures --zero_init --clipping\
        --mode_fwd subtb --lr_policy 1e-4 --lr_flow 1e-3 \
        --partial_energy --conditional_flow_model\
        --langevin --epochs 30000 \
        --exploratory --exploration_wd --exploration_factor 0.2 \
        --hidden_dim 128 --t_emb_dim 128 --batch_size 128
done