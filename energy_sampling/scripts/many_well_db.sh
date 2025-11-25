for seed in 0 1 2
do
    CUDA_VISIBLE_DEVICES=0, python train.py \
        --seed $seed \
        --t_scale 1. --energy many_well --pis_architectures --zero_init --clipping\
        --mode_fwd db --lr_policy 1e-3 --lr_flow 1e-2 \
        --conditional_flow_model \
        --exploratory --exploration_wd --exploration_factor 0.2 \

done