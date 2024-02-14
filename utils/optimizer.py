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

import torch.nn
from torch import optim
from typing import Iterator
from torch.nn.parameter import Parameter

import logging
logger = logging.getLogger("optimizer")


def get_optimizer(model_param: Iterator[Parameter], optimizer_config: dict) -> torch.optim.Optimizer:

    optimizer_name = optimizer_config['optimizer_name']
    learning_rate = optimizer_config['learning_rate']
    momentum = optimizer_config['momentum']
    weight_decay = optimizer_config['weight_decay']
    dampening = optimizer_config['dampening']
    nesterov = optimizer_config['nesterov']

    if optimizer_name == 'sgd':
        return optim.SGD(params=model_param, lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                         dampening=dampening, nesterov=nesterov)
    elif optimizer_name == 'adam':
        return optim.Adam(params=model_param, lr=learning_rate, weight_decay=weight_decay)
    else:
        logger.error(f"No implementation is available for {optimizer_name}!")
        logger.info("You can add the corresponding implementation to get_optimizer in utils/optimizer.py")
        logger.info("Or, you can use --optimizer sgd|adam")
        exit()

