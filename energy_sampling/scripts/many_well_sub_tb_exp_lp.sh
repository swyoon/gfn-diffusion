for seed in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0, python train.py \
        --seed $seed \
        --t_scale 1. --energy many_well --pis_architectures --zero_init --clipping\
        --mode_fwd subtb --lr_policy 1e-3 --lr_flow 1e-2 \
        --partial_energy --conditional_flow_model\
        --langevin --epochs 10000 \
        --exploratory --exploration_wd --exploration_factor 0.2 \

done