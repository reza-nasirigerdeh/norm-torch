# ######
# copy the imagenet dataset to ../datasets/

# BatchNorm-based ResNet-18/34/50
model_list=(resnet18_bn resnet34_bn resnet50_bn)
for model in "${model_list[@]}"
do
OMP_NUM_THREADS=64 NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=8 simulate_multi_gpu.py --model $model --data-path ../datasets/imagenet/
done

# GroupNorm-based ResNet-18/34/50
model_list=(resnet18_gn resnet34_gn resnet50_gn)
for model in "${model_list[@]}"
do
OMP_NUM_THREADS=64 NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=8 simulate_multi_gpu.py --model $model --data-path ../datasets/imagenet/
done

# LayerNorm-based ResNet-18/34/50
model_list=(resnet18_ln resnet34_ln resnet50_ln)
for model in "${model_list[@]}"
do
  OMP_NUM_THREADS=64 NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=8 simulate_multi_gpu.py --model $model --data-path ../datasets/imagenet/
done

# KNResNet-18/34/50
model_list=(knresnet18 knresnet34 knresnet50)
for model in "${model_list[@]}"
do
  OMP_NUM_THREADS=64 NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=8 simulate_multi_gpu.py --model $model --data-path ../datasets/imagenet/
done