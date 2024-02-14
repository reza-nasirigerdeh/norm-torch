# ######
# copy the imagenet dataset to ../datasets/

# LayerNorm based ConvNextTiny
OMP_NUM_THREADS=64 NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=4 simulate_multi_gpu.py \
--model convnext_tiny --batch-size 256 --opt adamw --lr 1e-3 --lr-scheduler cosineannealinglr \
--lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 600 --random-erase 0.1 \
--label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --weight-decay 0.05 --norm-weight-decay 0.0 \
--train-crop-size 176 --model-ema --val-resize-size 232 --ra-sampler --ra-reps 4 --data-path ../datasets/imagenet/