"""
    Copyright 2023 Reza NasiriGerdeh. All Rights Reserved.

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

from torch import optim

import logging
logger = logging.getLogger("optimizer")


def get_optimizer(optimizer_name, model_params, learning_rate, momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=False):
    if optimizer_name == 'sgd':
        return optim.SGD(params=model_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
                         dampening=dampening, nesterov=nesterov)
    elif optimizer_name == 'adam':
        return optim.Adam(params=model_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        logger.error(f"No implementation is available for {optimizer_name}!")
        logger.error("You can add the corresponding implementation to get_optimizer in utils/optimizer.py")
        logger.error("Or, you can use --optimizer sgd|adam")
        exit()

