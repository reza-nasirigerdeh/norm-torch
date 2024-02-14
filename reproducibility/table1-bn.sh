# ######
# ./run-cifar100.sh model_name learning_rate batch_size

# BatchNorm based resnet18/34/50 and preact_resnet18/34/50
# Batch size of 2
./run-cifar100.sh resnet18_bn 0.00078125 2
./run-cifar100.sh preact_resnet18_bn 0.00078125 2
./run-cifar100.sh resnet34_bn 0.00078125 2
./run-cifar100.sh preact_resnet34_bn 0.000390625 2
./run-cifar100.sh resnet50_bn 0.000390626 2
./run-cifar100.sh preact_resnet50_bn 0.000195313 2

# Batch size of 32
./run-cifar100.sh resnet18_bn 0.025 32
./run-cifar100.sh preact_resnet18_bn 0.025 32
./run-cifar100.sh resnet34_bn 0.025 32
./run-cifar100.sh preact_resnet34_bn 0.025 32
./run-cifar100.sh resnet50_bn 0.0125 32
./run-cifar100.sh preact_resnet50_bn 0.0125 32

# Batch size of 256
./run-cifar100.sh resnet18_bn 0.2 256
./run-cifar100.sh preact_resnet18_bn 0.2 256
./run-cifar100.sh resnet34_bn 0.1 256
./run-cifar100.sh preact_resnet34_bn 0.2 256
./run-cifar100.sh resnet50_bn 0.1 256
./run-cifar100.sh preact_resnet50_bn 0.2 256