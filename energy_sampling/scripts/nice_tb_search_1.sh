# Search: t_scale 0.03, 0.1, 0.3, 1, lr original & original/10
# Memory issue with hidden_dim 2048
# exploration_factor = t_scale * 0.2

CUDA_VISIBLE_DEVICES=1, python train.py \
    --seed 1\
    --hidden_dim 1024 --s_emb_dim 1024 --t_emb_dim 1024 --joint_layers 3\
    --t_scale 0.03 \
    --energy nice \
    --pis_architectures \
    --zero_init \
    --clipping \
    --mode_fwd tb \
    --lr_policy 1e-4 \
    --lr_flow 1e-2\
    --exploratory --exploration_wd --exploration_factor 0.006 \

CUDA_VISIBLE_DEVICES=1, python train.py \
    --seed 1\
    --hidden_dim 1024 --s_emb_dim 1024 --t_emb_dim 1024 --joint_layers 3\
    --t_scale 0.1 \
    --energy nice \
    --pis_architectures \
    --zero_init \
    --clipping \
    --mode_fwd tb \
    --lr_policy 1e-4 \
    --lr_flow 1e-2\
    --exploratory --exploration_wd --exploration_factor 0.02 \

CUDA_VISIBLE_DEVICES=1, python train.py \
    --seed 1\
    --hidden_dim 1024 --s_emb_dim 1024 --t_emb_dim 1024 --joint_layers 3\
    --t_scale 0.3 \
    --energy nice \
    --pis_architectures \
    --zero_init \
    --clipping \
    --mode_fwd tb \
    --lr_policy 1e-4 \
    --lr_flow 1e-2\
    --exploratory --exploration_wd --exploration_factor 0.06 \

CUDA_VISIBLE_DEVICES=1, python train.py \
    --seed 1\
    --hidden_dim 1024 --s_emb_dim 1024 --t_emb_dim 1024 --joint_layers 3\
    --t_scale 1 \
    --energy nice \
    --pis_architectures \
    --zero_init \
    --clipping \
    --mode_fwd tb \
    --lr_policy 1e-4 \
    --lr_flow 1e-2\
    --exploratory --exploration_wd --exploration_factor 0.2 \




