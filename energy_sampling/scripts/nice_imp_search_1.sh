# Search: t_scale 0.03, 0.1, 0.3, 1, lr original & original/10
# Memory issue with hidden_dim 2048
# exploration_factor = t_scale * 0.1
# ld_step is lowered to 0.0001 as initial acceptance rate is too low when using 0.1. 
# ld_step is lowered to 0.00001 in t_scale 0.03 & 0.1
# This is okay as ld_step is adjusted to reach the target acceptance rate of 0.574.

CUDA_VISIBLE_DEVICES=1, python train.py\
    --seed 1\
    --hidden_dim 1024 --s_emb_dim 1024 --t_emb_dim 1024 --joint_layers 3\
    --t_scale 1 --energy nice --pis_architectures --zero_init --clipping\
    --mode_fwd tb-avg --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
    --exploratory --exploration_wd --exploration_factor 0.1\
    --both_ways --local_search\
    --buffer_size 600000 --prioritized rank --rank_weight 0.01\
    --ld_step 0.0001 --ld_schedule --target_acceptance_rate 0.574\
    --langevin --epochs 10000

CUDA_VISIBLE_DEVICES=1, python train.py\
    --seed 1\
    --hidden_dim 1024 --s_emb_dim 1024 --t_emb_dim 1024 --joint_layers 3\
    --t_scale 0.3 --energy nice --pis_architectures --zero_init --clipping\
    --mode_fwd tb-avg --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
    --exploratory --exploration_wd --exploration_factor 0.03\
    --both_ways --local_search\
    --buffer_size 600000 --prioritized rank --rank_weight 0.01\
    --ld_step 0.0001 --ld_schedule --target_acceptance_rate 0.574\
    --langevin --epochs 10000

CUDA_VISIBLE_DEVICES=1, python train.py\
    --seed 1\
    --hidden_dim 1024 --s_emb_dim 1024 --t_emb_dim 1024 --joint_layers 3\
    --t_scale 0.1 --energy nice --pis_architectures --zero_init --clipping\
    --mode_fwd tb-avg --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
    --exploratory --exploration_wd --exploration_factor 0.01\
    --both_ways --local_search\
    --buffer_size 600000 --prioritized rank --rank_weight 0.01\
    --ld_step 0.00001 --ld_schedule --target_acceptance_rate 0.574\
    --langevin --epochs 10000

CUDA_VISIBLE_DEVICES=1, python train.py\
    --seed 1\
    --hidden_dim 1024 --s_emb_dim 1024 --t_emb_dim 1024 --joint_layers 3\
    --t_scale 0.03 --energy nice --pis_architectures --zero_init --clipping\
    --mode_fwd tb-avg --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
    --exploratory --exploration_wd --exploration_factor 0.003\
    --both_ways --local_search\
    --buffer_size 600000 --prioritized rank --rank_weight 0.01\
    --ld_step 0.00001 --ld_schedule --target_acceptance_rate 0.574\
    --langevin --epochs 10000


