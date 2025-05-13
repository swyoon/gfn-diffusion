CUDA_VISIBLE_DEVICES=1, python train.py \
    --seed 0 \
    --t_scale 1. --energy hard_funnel --pis_architectures --zero_init --clipping\
    --mode_fwd db --lr_policy 1e-3 --lr_flow 1e-2 \
    --conditional_flow_model\
    --exploratory --exploration_wd --exploration_factor 0.2 \
