# norm-torch
**norm-torch** is a PyTorch-based simulation framework to simulate popular convolution neural networks such as **ResNets** and 
**VGGNets** using different normalization layers including **BatchNorm**, **LayerNorm**, **GroupNorm**, and **NoNorm** as baseline.

**The code of KernelNorm, our proposed normalization layer, is coming soon.**

# Installation
Clone the NormTorch repository:
```
git clone https://github.com/reza-nasirigerdeh/norm-torch
```
Install the dependencies:
```
pip3 install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
```

# Options
### Dataset
| Dataset              | option                  |
|:---------------------|:------------------------|
| MNIST                | --dataset mnist         |
| Fashion-MNIST        | --dataset fashion_mnist |
| CIFAR-10             | --dataset cifar10       |
| CIFAR-100            | --dataset cifar100      |
| Imagenette 160-pixel | --dataset imagenette_160px |
| Imagenette 320-pixel | --dataset imagenette_320px |
| Imagewoof 160-pixel  | --dataset imagewoof_160px |
| Imagewoof 320-pixel  | --dataset imagewoof_320px |

### Preprocessing
| operation                                                          | option                           |
|:-------------------------------------------------------------------|:---------------------------------|
| resize train images (e.g. to 192x192)                              | --resize-train 192x192           |
| resize test images (e.g. to 192x192)                               | --resize-test 192x192            |
| random horizontal flip                                             | --random-hflip                   |
| random cropping with given size e.g. 32x32 and padding e.g. 4x4    | --random-crop 32x32-4x4          |
| random resized crop with given size (e.g. 128x128)                 | --random-resized-crop 128x128    |
| center crop test images with given size (e.g. 128x128)             | --center-crop 128x128            |
| normalize with mean (e.g. with 0.4625,0.4580,0.4298)               | --norm-mean 0.4625,0.4580,0.4298 |
| normalize with standard-deviation (e.g. with 0.2786,0.2755,0.2982) | --norm-std 0.2786,0.2755,0.2982  |

### Loss function
| Loss function | option               |
|:--------------|:---------------------|
| Cross-entropy | --loss cross_entropy |

### Optimizer
#### Stochastic gradient descent (SGD)
| Optimizer                                      | option                             |
|:-----------------------------------------------|:-----------------------------------|
| SGD                                            | --optimizer sgd                    |
| learning rate (e.g. 0.01)                      | --learning-rate  0.01              |
| momentum (e.g. 0.9)                            | --momentum 0.9                     |
| weight decay (e.g. 0.0001)                     | --weight-decay 0.0001              |
| dampening (e.g. 0.0)                           | --dampening 0.0                    |
| Nesterov momentum                              | --nesterov                         |

#### Adam
| Optimizer                 | option                |
|:--------------------------|:----------------------|
| Adam                      | --optimizer adam      |
| learning rate (e.g. 0.01) | --learning-rate  0.01 |

#### Learning rate scheduler
| Scheduler                                                               | option                           |
|:------------------------------------------------------------------------|:---------------------------------|
| multi-step learning rate scheduler                                      | --lr-scheduler  multi_step       |
| cosine-annealing learning rate scheduler                                | --lr-scheduler  cosine_annealing |
| decrease learning rate at epochs (e.g. 50, 75) for multi-step scheduler | --decay-epochs 50,75             |
| decay learning rate by factor of e.g. 10                                | --decay-multiplier 0.1           |

### Model
#### Toy models
| Model                                                      | option                                 |
|:-----------------------------------------------------------|:---------------------------------------|
| simple fully-connected model                               | --model fnn                            |
| simple convolutional model                                 | --model cnn                            |

#### VGG-6
| Model                                                | option                          |
|:-----------------------------------------------------|:--------------------------------|
| VGG-6 with no normalization layer                    | --model vgg6_nn                 |
| VGG-6 with batch normalization                       | --model vgg6_bn                 |
| VGG-6 with layer normalization                       | --model vgg6_ln                 |
| VGG-6 with group normalization of group size e.g. 32 | --model vgg6_gn --group-size 32 |

#### ResNet18/34/50/101/152
| Model                                                   | option                              |
|:--------------------------------------------------------|:------------------------------------|
| ResNet18 with no normalization                          | --model resnet18_nn                 |
| ResNet18 with batch normalization                       | --model resnet18_bn                 |
| ResNet18 with layer normalization                       | --model resnet18_ln                 |
| ResNet18 with group normalization of group size e.g. 32 | --model resnet18_gn --group-size 32 |

#### PreactResNet18/34/50/101/152
| Model                                                         | option                                     |
|:--------------------------------------------------------------|:-------------------------------------------|
| PreactResNet18 with no normalization                          | --model preact_resnet18_nn                 |
| PreactResNet18 with batch normalization                       | --model preact_resnet18_bn                 |
| PreactResNet18 with layer normalization                       | --model preact_resnet18_ln                 |
| PreactResNet18 with group normalization of group size e.g. 32 | --model preact_resnet18_gn --group-size 32 |

#### Note: For the other versions of ResNet and PreactResNet, please specify the corresponding version number instead of 18. 
**Example2**: ResNet50 with group normalization --> --model resnet50_gn \
**Example3**: PreactResNet34 with layer normalization --> --model preact_resnet34_ln 

### Other
| Description                                                     | option                     |
|:----------------------------------------------------------------|:---------------------------|
| batch size (e.g. 32)                                            | --batch-size 32            |
| number of train epochs (e.g. 100)                               | --epochs 100               |
| run number to have a separate result file for each run (e.g. 1) | --run 1                    |
| log level (e.g. debug)                                          | --log-level debug          |
# Run
**Example1**: Train batch normalized version of VGG-6 on CIFAR-10 with cross-entropy loss function, SGD optimizer with learning rate of 0.025 and momentum of 0.9, and batch size of 32 for 100 epochs:

```
python3 simulate.py --dataset cifar10 --model vgg6_bn --optimizer sgd --momentum 0.9 \
                    --loss cross_entropy --learning-rate 0.025 --batch-size 32 --epochs 100
```

**Example2**: Train layer normalized version of ResNet-34 on imagenette-160-pixel with SGD optimizer, learning rate of 0.005 and momentum of 0.9, and batch size of 16 for 100 epochs. For preprocessing, apply random-resized-crop of shape 128x128 and random horizontal flipping to the train images, and resize test images to 160x160 first, and then, center crop them to 128x128, finally normalize them with the mean and std of imagenet:

```
python3 simulate.py --dataset imagenette_160px --random-hflip \
                    --random-resized-crop 128x128 --resize-test 160x160 --center-crop 128x128  \
                    --norm-mean 0.485,0.456,0.406  --norm-std 0.229,0.224,0.225 \
                    --model resnet34_ln --optimizer sgd --momentum 0.9 \
                    --learning-rate 0.005 --batch-size 16 --epochs 100
```

**Example3**: Train group normalized version of PreactResNet-18 with group size of 32 on CIFAR-100 with SGD optimizer, learning rate of 0.01, momentum of 0.9, and batch size of 16 for 200 epochs. For preprocessing, apply random horizontal flipping and cropping with padding 4x4 to the images. Decay the learning rate by factor of 10 at epochs 100 and 150.

```
python3 simulate.py --dataset cifar100 --random-crop 32x32-4x4 \
                    --random-hflip --model preact_resnet18_gn --group-size 32  \
                    --optimizer sgd --momentum 0.9  --learning-rate 0.01 \
                    --multistep-lr-scheduler --decay-epochs 100,150 --decay-multiplier 0.1 \
                    --batch-size 16 --epochs 200
```

## Citation
If you use **norm-torch** in your study, please cite the following paper: <br />
   ```
@misc{nasirigerdeh2022-kernelnorm,
       title={Kernel Normalized Convolutional Networks},
       author={Reza Nasirigerdeh, Reihaneh Torkzadehmahani, Daniel Rueckert, Georgios Kaissis},
       year={2022},
       eprint={2205.10089},
       archivePrefix={arXiv},
       primaryClass={cs.LG cs.CV},
       howpublished = "\url{https://arxiv.org/abs/2205.10089}",
}
   ```