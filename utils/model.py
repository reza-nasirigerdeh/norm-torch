"""
    MIT License
    Copyright (c) 2024 Reza NasiriGerdeh

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

# import ResNet models
from models.resnet.resnet_nn import resnet18_nn, resnet34_nn, resnet50_nn
from models.resnet.resnet_bn import resnet18_bn, resnet34_bn, resnet50_bn
from models.resnet.resnet_ln import resnet18_ln, resnet34_ln, resnet50_ln
from models.resnet.resnet_gn import resnet18_gn, resnet34_gn, resnet50_gn

# import PreactResNet models
from models.preact_resnet.preact_resnet_nn import preact_resnet18_nn, preact_resnet34_nn, preact_resnet50_nn
from models.preact_resnet.preact_resnet_bn import preact_resnet18_bn, preact_resnet34_bn, preact_resnet50_bn
from models.preact_resnet.preact_resnet_ln import preact_resnet18_ln, preact_resnet34_ln, preact_resnet50_ln
from models.preact_resnet.preact_resnet_gn import preact_resnet18_gn, preact_resnet34_gn, preact_resnet50_gn

# import KNResNet models
from models.knresnet.knresnet import knresnet18, knresnet34, knresnet50

import torch
from models.toynet import CNN


import logging
logger = logging.getLogger("model")


def build_model(model_config: dict) -> torch.nn.Module:

    model_name = model_config['model_name']
    num_classes = model_config['num_classes']
    num_groups = model_config['num_groups']
    low_resolution = model_config['low_resolution']
    kn_dropout_p = model_config['kn_dropout_p']

    logger.info(f"Building model {model_name} ...")

    # #### Simple convolutional model
    if model_name == 'cnn':
        model = CNN(num_classes=num_classes)

    # #### ResNet-18
    elif model_name == 'resnet18_nn':
        model = resnet18_nn(num_classes=num_classes, low_resolution=low_resolution)

    elif model_name == 'resnet18_bn':
        model = resnet18_bn(num_classes=num_classes, low_resolution=low_resolution)

    elif model_name == 'resnet18_gn':
        model = resnet18_gn(num_classes=num_classes, num_groups=num_groups, low_resolution=low_resolution)

    elif model_name == 'resnet18_ln':
        model = resnet18_ln(num_classes=num_classes, low_resolution=low_resolution)

    # #### ResNet-34
    elif model_name == 'resnet34_nn':
        model = resnet34_nn(num_classes=num_classes, low_resolution=low_resolution)

    elif model_name == 'resnet34_bn':
        model = resnet34_bn(num_classes=num_classes, low_resolution=low_resolution)

    elif model_name == 'resnet34_gn':
        model = resnet34_gn(num_classes=num_classes, num_groups=num_groups, low_resolution=low_resolution)

    elif model_name == 'resnet34_ln':
        model = resnet34_ln(num_classes=num_classes, low_resolution=low_resolution)

    # #### ResNet-50
    elif model_name == 'resnet50_nn':
        model = resnet50_nn(num_classes=num_classes, low_resolution=low_resolution)

    elif model_name == 'resnet50_bn':
        model = resnet50_bn(num_classes=num_classes, low_resolution=low_resolution)

    elif model_name == 'resnet50_gn':
        model = resnet50_gn(num_classes=num_classes, num_groups=num_groups, low_resolution=low_resolution)

    elif model_name == 'resnet50_ln':
        model = resnet50_ln(num_classes=num_classes, low_resolution=low_resolution)

    # #### PreactResNet-18
    elif model_name == 'preact_resnet18_nn':
        model = preact_resnet18_nn(num_classes=num_classes, low_resolution=low_resolution)

    elif model_name == 'preact_resnet18_bn':
        model = preact_resnet18_bn(num_classes=num_classes, low_resolution=low_resolution)

    elif model_name == 'preact_resnet18_gn':
        model = preact_resnet18_gn(num_classes=num_classes, num_groups=num_groups, low_resolution=low_resolution)

    elif model_name == 'preact_resnet18_ln':
        model = preact_resnet18_ln(num_classes=num_classes, low_resolution=low_resolution)

    # #### PreactResNet-34
    elif model_name == 'preact_resnet34_nn':
        model = preact_resnet34_nn(num_classes=num_classes, low_resolution=low_resolution)

    elif model_name == 'preact_resnet34_bn':
        model = preact_resnet34_bn(num_classes=num_classes, low_resolution=low_resolution)

    elif model_name == 'preact_resnet34_gn':
        model = preact_resnet34_gn(num_classes=num_classes, num_groups=num_groups, low_resolution=low_resolution)

    elif model_name == 'preact_resnet34_ln':
        model = preact_resnet34_ln(num_classes=num_classes, low_resolution=low_resolution)

    # #### PreactResNet-50
    elif model_name == 'preact_resnet50_nn':
        model = preact_resnet50_nn(num_classes=num_classes, low_resolution=low_resolution)

    elif model_name == 'preact_resnet50_bn':
        model = preact_resnet50_bn(num_classes=num_classes, low_resolution=low_resolution)

    elif model_name == 'preact_resnet50_gn':
        model = preact_resnet50_gn(num_classes=num_classes, num_groups=num_groups, low_resolution=low_resolution)

    elif model_name == 'preact_resnet50_ln':
        model = preact_resnet50_ln(num_classes=num_classes, low_resolution=low_resolution)

    # ##### KNResNets
    elif model_name == 'knresnet18':
        model = knresnet18(num_classes=num_classes, dropout_p=kn_dropout_p, low_resolution=low_resolution)

    elif model_name == 'knresnet34':
        model = knresnet34(num_classes=num_classes, dropout_p=kn_dropout_p, low_resolution=low_resolution)

    elif model_name == 'knresnet50':
        model = knresnet50(num_classes=num_classes, dropout_p=kn_dropout_p, low_resolution=low_resolution)

    else:
        logger.error(f'{model_name} is an valid model name!')
        logger.info(f'You can add the model definition to build_model in utils/model.py')
        logger.info("Exiting ....")
        exit()

    logger.info(model)
    print()

    return model
