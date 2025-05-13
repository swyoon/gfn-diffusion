# tb best
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

# Subtb best
CUDA_VISIBLE_DEVICES=0, python train.py \
    --seed 0\
    --hidden_dim 512 --s_emb_dim 512 --t_emb_dim 512 --joint_layers 3\
    --t_scale 0.03 --energy nice --pis_architectures --zero_init --clipping\
    --mode_fwd subtb --lr_policy 1e-4 --lr_flow 1e-3 \
    --partial_energy --conditional_flow_model\
    --langevin --epochs 10000 \
    --exploratory --exploration_wd --exploration_factor 0.006 \

# imp best
CUDA_VISIBLE_DEVICES=0, python train.py\
    --seed 0\
    --hidden_dim 1024 --s_emb_dim 1024 --t_emb_dim 1024 --joint_layers 3\
    --t_scale 0.03 --energy nice --pis_architectures --zero_init --clipping\
    --mode_fwd tb-avg --lr_policy 1e-4 --lr_back 1e-4 --lr_flow 1e-2\
    --exploratory --exploration_wd --exploration_factor 0.003\
    --both_ways --local_search\
    --buffer_size 600000 --prioritized rank --rank_weight 0.01\
    --ld_step 0.00001 --ld_schedule --target_acceptance_rate 0.574\
    --langevin --epochs 10000