for T in 1000 800 600 400 200 100
do
    CUDA_VISIBLE_DEVICES=0, python train.py \
        --seed 0 \
        --T $T \
        --t_scale 1. --energy hard_funnel --pis_architectures --zero_init --clipping\
        --mode_fwd subtb --lr_policy 1e-3 --lr_flow 1e-2 \
        --conditional_flow_model\

done
