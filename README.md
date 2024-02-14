# NormTorch
**NormTorch** is a PyTorch-based simulation framework to compare the performance of the existing 
normalization layers including **BatchNorm**, **LayerNorm**, **GroupNorm**, and our proposed **KernelNorm** in
image classification with popular convolutional neural networks such as **ResNets**.

# Requirements
- Python +3.9
- PyTorch +2.0.1

# Installation
Clone the norm-torch repository:
```
git clone https://github.com/reza-nasirigerdeh/norm-torch
```
Install the dependencies:
```
pip3 install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
```

# Run
**Single GPU**: Train ResNets and PreactResNets on CIFAR-10/100 

Example 1: Train **batch normalized** ResNet-18 on CIFAR-100
```
python3 simulate.py --dataset cifar100 --low-resolution --num-classes 100 --model resnet18_bn \
                    --optimizer sgd --momentum 0.9 --learning-rate 0.025 --batch-size 32 --epochs 150 \
                    --random-hflip --random-crop 32x32-4x4 \
                    --norm-mean 0.5071,0.4865,0.4409  --norm-std 0.2673,0.2564,0.2762 \
                    --loss cross_entropy  --weight-decay 0.0005 \
                    --lr-scheduler cosine_annealing --decay-epochs 150 --decay-multiplier 0.01 \
                    --checkpoint-freq 1 --run 1
```
In general, to train different versions of ResNets or PreactResNets using BatchNorm/GroupNorm/LayerNorm, you can use: </br>
resnet18_bn | resnet34_bn | resnet50_bn </br>
resnet18_gn | resnet34_gn | resnet50_gn </br>
resnet18_ln | resnet34_ln | resnet50_ln </br>
preact_resnet18_bn | preact_resnet34_bn | preact_resnet50_bn </br>
preact_resnet18_gn | preact_resnet34_gn | preact_resnet50_gn </br>
preact_resnet18_ln | preact_resnet34_ln | preact_resnet50_ln </br>

Example 2: Train **kernel normalized** ResNet-34 on CIFAR-100
```
python3 simulate.py --dataset cifar100 --low-resolution --num-classes 100 --model knresnet34 \
                    --optimizer sgd --momentum 0.9 --learning-rate 0.05 --batch-size 32 --epochs 150 \
                    --random-hflip --random-crop 32x32-4x4 \
                    --loss cross_entropy  --weight-decay 0.0005 \
                    --lr-scheduler cosine_annealing --decay-epochs 150 --decay-multiplier 0.01 \
                    --checkpoint-freq 1 --run 1
```
Note: For KNResNets, we do NOT normalize the samples using the mean and std of the dataset unlike other normalization methods. </br>
To train a specific version of KNResNets, you can use: knresnet18 | knresnet34 | knresnet50 

Example 3: Train **group normalized** ResNet-18 on CIFAR-10
```
python3 simulate.py --dataset cifar10 --low-resolution --num-classes 10 --model resnet18_gn \
                    --optimizer sgd --momentum 0.9 --learning-rate 0.0125 --batch-size 32 --epochs 75 \
                    --random-hflip --random-crop 32x32-4x4 \
                    --norm-mean 0.4914,0.4822,0.4465  --norm-std 0.2471,0.2435,0.2616 \
                    --loss cross_entropy \
                    --lr-scheduler cosine_annealing --decay-epochs 75 --decay-multiplier 0.01 \
                    --checkpoint-freq 1 --run 1
```

Example 4: Train **kernel normalized** ResNet-18 on CIFAR-10
```
python3 simulate.py --dataset cifar10 --low-resolution --num-classes 10 --model knresnet18 \
                    --optimizer sgd --momentum 0.9 --learning-rate 0.05 --batch-size 32 --epochs 75 \
                    --random-hflip --random-crop 32x32-4x4 \
                    --loss cross_entropy \
                    --lr-scheduler cosine_annealing --decay-epochs 75 --decay-multiplier 0.01 \
                    --checkpoint-freq 1 --run 1
```

**Multi-GPU**: Train ResNets and ConvNextTiny on ImageNet <br /> <br />
Example 1: Train **batch normalized** ResNet-50 on ImageNet using 8 GPUs
```
OMP_NUM_THREADS=64 NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=8 simulate_multi_gpu.py \
                                      --model resnet50_bn --data-path ./datasets/imagenet/
```
Note that the path of the ImageNet dataset should be given to --data-path. <br />

Example 2: Train **kernel normalized** ResNet-34 on ImageNet using 8 GPUs
```
OMP_NUM_THREADS=64 NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=8 simulate_multi_gpu.py \
                                      --model knresnet34 --data-path ./datasets/imagenet/
```

## Reproducibility
To reproduce the results in the KernelNorm paper, please follow the below instructions: </br >
### Image classification (Table1, Table2, and ConvNextTiny)
The reproducibility bash scripts have been provided in the ["reproducibility"](reproducibility) folder of this repo.

### Semantic segmentation (Table3)
The corresponding models have been provided in the ["segmentation"](models/segmentation) directory of this repo.
To reproduce the results in the paper, please copy the segmentation models from this repo to the "lib/models" folder in the following repository: </br >
https://github.com/HRNet/HRNet-Semantic-Segmentation

Next, train the model using the "tools/train.py" script in the aforementioned repo. 

### Differentially private image classification (Table4)
Please use our another repository dedicated to differentially private learning: </br >
https://github.com/reza-nasirigerdeh/dp-torch

## Pretrained KNResNets
The weights of the KNResNet18/34/50 models trained on ImageNet is available in the following: </br >

| Model                    | Top-1 Accuracy | Weights                                                                                                       |
|:-------------------------|:---------------|:--------------------------------------------------------------------------------------------------------------|
| KNResNet-18             | 71.17%         | [knresnet18_imagenet.pth](https://drive.google.com/file/d/1oU4IGxErW4l-oqY6vn8V1DL0dBMt1KUO/view?usp=sharing) |
| KNResNet-34             | 74.60%         | [knresnet34_imagenet.pth](https://drive.google.com/file/d/1dn1O_JHcAP_6gQgvSe5ojV_WzDOB7Lvk/view?usp=sharing) |
| KNResNet-50             | 76.54%         | [knresnet50_imagenet.pth](https://drive.google.com/file/d/1CSP4HQTQWaR0q2Pdf4GyPfMj6TDE6E2J/view?usp=sharing)                                                                                       |


## Citation
If you use **norm-torch** in your study, please cite the KernelNorm paper: <br />
   ```
@article{
    nasirigerdeh2024kernelnorm,
    title={Kernel Normalized Convolutional Networks},
    author={Reza Nasirigerdeh and Reihaneh Torkzadehmahani and Daniel Rueckert and Georgios Kaissis},
    journal={Transactions on Machine Learning Research},
    year={2024},
    url={https://openreview.net/forum?id=Uv3XVAEgG6},
}
   ```