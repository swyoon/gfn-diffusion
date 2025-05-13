for seed in 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0, python train.py\
    --seed $seed\
    --hidden_dim 1024 --s_emb_dim 1024 --t_emb_dim 1024 --joint_layers 3\
    --t_scale 0.03 --energy nice --pis_architectures --zero_init --clipping\
    --mode_fwd tb-avg --lr_policy 1e-4 --lr_back 1e-4 --lr_flow 1e-2\
    --exploratory --exploration_wd --exploration_factor 0.003\
    --both_ways --local_search\
    --buffer_size 600000 --prioritized rank --rank_weight 0.01\
    --ld_step 0.00001 --ld_schedule --target_acceptance_rate 0.574\
    --langevin --epochs 10000
done