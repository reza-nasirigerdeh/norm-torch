# ######
# ./run-cifar100.sh model_name learning_rate batch_size

# KernelNorm based resnet18/34/50
# Batch size of 2
./run-cifar100.sh knresnet18 0.0015625 2
./run-cifar100.sh knresnet34 0.0015625 2
./run-cifar100.sh knresnet50 0.0015625 2

# Batch size of 32
./run-cifar100.sh knresnet18 0.05 32
./run-cifar100.sh knresnet34 0.05 32
./run-cifar100.sh knresnet50 0.025 32

# Batch size of 256
./run-cifar100.sh knresnet18 0.2 256
./run-cifar100.sh knresnet34 0.2 256
./run-cifar100.sh knresnet50 0.2 256
