# ######
# ./run-cifar100.sh model_name learning_rate batch_size

# LayerNorm based resnet18/34/50 and preact_resnet18/34/50
# Batch size of 2
./run-cifar100.sh resnet18_ln 0.0015625 2
./run-cifar100.sh preact_resnet18_ln 0.0015625 2
./run-cifar100.sh resnet34_ln 0.0015625 2
./run-cifar100.sh preact_resnet34_ln 0.0015625 2
./run-cifar100.sh resnet50_ln 0.00078125 2
./run-cifar100.sh preact_resnet50_ln 0.0015625 2

# Batch size of 32
./run-cifar100.sh resnet18_ln 0.0125 32
./run-cifar100.sh preact_resnet18_ln 0.0125 32
./run-cifar100.sh resnet34_ln 0.0125 32
./run-cifar100.sh preact_resnet34_ln 0.0125 32
./run-cifar100.sh resnet50_ln 0.0125 32
./run-cifar100.sh preact_resnet50_ln 0.0125 32

# Batch size of 256
./run-cifar100.sh resnet18_ln 0.05 256
./run-cifar100.sh preact_resnet18_ln 0.05 256
./run-cifar100.sh resnet34_ln 0.05 256
./run-cifar100.sh preact_resnet34_ln 0.05 256
./run-cifar100.sh resnet50_ln 0.05 256
./run-cifar100.sh preact_resnet50_ln 0.05 256