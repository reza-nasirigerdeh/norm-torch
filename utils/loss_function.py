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

from torch import nn

import logging
logger = logging.getLogger("loss_function")


def get_loss_function(loss_function_name):
    if loss_function_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        logger.error(f"No implementation is available for {loss_function_name}!")
        logger.error("You can add the corresponding implementation to get_loss_function in utils/loss_function.py")
        logger.error("Or, you can use --loss cross_entropy")
        exit()
