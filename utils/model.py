"""
    Copyright 2023 Reza NasiriGerdeh and Javad TorkzadehMahani. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import logging
from torchvision import models
from models.toynet import FNN, CNN

from models.vgg.vgg6 import vgg6_nn, vgg6_bn, vgg6_gn, vgg6_ln

from models.resnet.resnet_nn import resnet18_nn, resnet34_nn, resnet50_nn, resnet101_nn, resnet152_nn
from models.resnet.resnet_gn import resnet18_gn, resnet34_gn, resnet50_gn, resnet101_gn, resnet152_gn
from models.resnet.resnet_ln import resnet18_ln, resnet34_ln, resnet50_ln, resnet101_ln, resnet152_ln


from models.preact_resnet.preact_resnet_bn import preact_resnet18_bn, preact_resnet34_bn, preact_resnet50_bn, preact_resnet101_bn, preact_resnet152_bn
from models.preact_resnet.preact_resnet_nn import preact_resnet18_nn, preact_resnet34_nn, preact_resnet50_nn, preact_resnet101_nn, preact_resnet152_nn
from models.preact_resnet.preact_resnet_ln import preact_resnet18_ln, preact_resnet34_ln, preact_resnet50_ln, preact_resnet101_ln, preact_resnet152_ln
from models.preact_resnet.preact_resnet_gn import preact_resnet18_gn, preact_resnet34_gn, preact_resnet50_gn, preact_resnet101_gn, preact_resnet152_gn


logger = logging.getLogger("model")


def build_model(model_name, num_classes, group_size=32):

    logger.info(f"Building model {model_name} ...")

    # ##### Simple fully-connected model
    if model_name == 'fnn':
        model = FNN(num_classes=num_classes)

    # ##### Simple convolutional model
    elif model_name == 'cnn':
        model = CNN(num_classes=num_classes)

    # ##### VGG-6
    elif model_name == 'vgg6_nn':
        model = vgg6_nn(num_classes=num_classes)
    elif model_name == 'vgg6_bn':
        model = vgg6_bn(num_classes=num_classes)
    elif model_name == 'vgg6_gn':
        model = vgg6_gn(num_classes=num_classes, group_size=group_size)
    elif model_name == 'vgg6_ln':
        model = vgg6_ln(num_classes=num_classes)

    # ##### ResNet18
    elif model_name == 'resnet18_nn':
        model = resnet18_nn(num_classes=num_classes)
    elif model_name == 'resnet18_bn':
        model = models.resnet18(num_classes=num_classes)
    elif model_name == 'resnet18_gn':
        model = resnet18_gn(num_classes=num_classes, group_size=group_size)
    elif model_name == 'resnet18_ln':
        model = resnet18_ln(num_classes=num_classes)

    # ##### ResNet34
    elif model_name == 'resnet34_nn':
        model = resnet34_nn(num_classes=num_classes)
    elif model_name == 'resnet34_bn':
        model = models.resnet34(num_classes=num_classes)
    elif model_name == 'resnet34_gn':
        model = resnet34_gn(num_classes=num_classes, group_size=group_size)
    elif model_name == 'resnet34_ln':
        model = resnet34_ln(num_classes=num_classes)

    # ##### ResNet50
    elif model_name == 'resnet50_nn':
        model = resnet50_nn(num_classes=num_classes)
    elif model_name == 'resnet50_bn':
        model = models.resnet50(num_classes=num_classes)
    elif model_name == 'resnet50_gn':
        model = resnet50_gn(num_classes=num_classes, group_size=group_size)
    elif model_name == 'resnet50_ln':
        model = resnet50_ln(num_classes=num_classes)

    # ##### ResNet101
    elif model_name == 'resnet101_nn':
        model = resnet101_nn(num_classes=num_classes)
    elif model_name == 'resnet101_bn':
        model = models.resnet101(num_classes=num_classes)
    elif model_name == 'resnet101_gn':
        model = resnet101_gn(num_classes=num_classes, group_size=group_size)
    elif model_name == 'resnet101_ln':
        model = resnet101_ln(num_classes=num_classes)

    # ##### ResNet152
    elif model_name == 'resnet152_nn':
        model = resnet152_nn(num_classes=num_classes)
    elif model_name == 'resnet152_bn':
        model = models.resnet152(num_classes=num_classes)
    elif model_name == 'resnet152_gn':
        model = resnet152_gn(num_classes=num_classes, group_size=group_size)
    elif model_name == 'resnet152_ln':
        model = resnet152_ln(num_classes=num_classes)

    # ##### PreactResNet18
    elif model_name == 'preact_resnet18_nn':
        model = preact_resnet18_nn(num_classes=num_classes)
    elif model_name == 'preact_resnet18_bn':
        model = preact_resnet18_bn(num_classes=num_classes)
    elif model_name == 'preact_resnet18_gn':
        model = preact_resnet18_gn(num_classes=num_classes, group_size=group_size)
    elif model_name == 'preact_resnet18_ln':
        model = preact_resnet18_ln(num_classes=num_classes)

    # ##### PreactResNet34
    elif model_name == 'preact_resnet34_nn':
        model = preact_resnet34_nn(num_classes=num_classes)
    elif model_name == 'preact_resnet34_bn':
        model = preact_resnet34_bn(num_classes=num_classes)
    elif model_name == 'preact_resnet34_gn':
        model = preact_resnet34_gn(num_classes=num_classes, group_size=group_size)
    elif model_name == 'preact_resnet34_ln':
        model = preact_resnet34_ln(num_classes=num_classes)

    # ##### PreactResNet50
    elif model_name == 'preact_resnet50_nn':
        model = preact_resnet50_nn(num_classes=num_classes)
    elif model_name == 'preact_resnet50_bn':
        model = preact_resnet50_bn(num_classes=num_classes)
    elif model_name == 'preact_resnet50_gn':
        model = preact_resnet50_gn(num_classes=num_classes, group_size=group_size)
    elif model_name == 'preact_resnet50_ln':
        model = preact_resnet50_ln(num_classes=num_classes)

    # ##### PreactResNet101
    elif model_name == 'preact_resnet101_nn':
        model = preact_resnet101_nn(num_classes=num_classes)
    elif model_name == 'preact_resnet101_bn':
        model = preact_resnet101_bn(num_classes=num_classes)
    elif model_name == 'preact_resnet101_gn':
        model = preact_resnet101_gn(num_classes=num_classes, group_size=group_size)
    elif model_name == 'preact_resnet101_ln':
        model = preact_resnet101_ln(num_classes=num_classes)

    # ##### PreactResNet152
    elif model_name == 'preact_resnet152_nn':
        model = preact_resnet152_nn(num_classes=num_classes)
    elif model_name == 'preact_resnet152_bn':
        model = preact_resnet152_bn(num_classes=num_classes)
    elif model_name == 'preact_resnet152_gn':
        model = preact_resnet152_gn(num_classes=num_classes, group_size=group_size)
    elif model_name == 'preact_resnet152_ln':
        model = preact_resnet152_ln(num_classes=num_classes)

    else:
        print(f'{model_name} is an valid model name!')
        print("Exiting ....")
        exit()

    return model
