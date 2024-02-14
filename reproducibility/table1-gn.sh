# ######
# ./run-cifar100.sh model_name learning_rate batch_size

# GroupNorm based resnet18/34/50 and preact_resnet18/34/50
# Batch size of 2
./run-cifar100.sh resnet18_gn 0.0015625 2
./run-cifar100.sh preact_resnet18_gn 0.0015625 2
./run-cifar100.sh resnet34_gn 0.0015625 2
./run-cifar100.sh preact_resnet34_gn 0.0015625 2
./run-cifar100.sh resnet50_gn 0.00078125 2
./run-cifar100.sh preact_resnet50_gn 0.0015625 2

# Batch size of 32
./run-cifar100.sh resnet18_gn 0.025 32
./run-cifar100.sh preact_resnet18_gn 0.025 32
./run-cifar100.sh resnet34_gn 0.025 32
./run-cifar100.sh preact_resnet34_gn 0.025 32
./run-cifar100.sh resnet50_gn 0.0125 32
./run-cifar100.sh preact_resnet50_gn 0.025 32

# Batch size of 256
./run-cifar100.sh resnet18_gn 0.1 256
./run-cifar100.sh preact_resnet18_gn 0.1 256
./run-cifar100.sh resnet34_gn 0.1 256
./run-cifar100.sh preact_resnet34_gn 0.1 256
./run-cifar100.sh resnet50_gn 0.05 256
./run-cifar100.sh preact_resnet50_gn 0.1 256