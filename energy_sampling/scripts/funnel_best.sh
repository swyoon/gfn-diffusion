for seed in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0, python train.py\
    --seed $seed\
    --t_scale 1. --energy hard_funnel --pis_architectures --zero_init --clipping\
    --mode_fwd tb --lr_policy 1e-3 --lr_back 1e-3 --lr_flow 1e-1\
    --exploratory --exploration_wd --exploration_factor 0.1\
    --both_ways --local_search\
    --buffer_size 600000 --prioritized rank --rank_weight 0.01\
    --ld_step 0.1 --ld_schedule --target_acceptance_rate 0.574\
    --langevin --epochs 10000
done