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

from typing import Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

import logging
logger = logging.getLogger("lr_scheduler")


def get_lr_scheduler(optimizer: Optimizer, train_config: dict) -> Tuple[MultiStepLR, CosineAnnealingLR]:
    lr_scheduler_name = train_config['lr_scheduler_name']
    decay_multiplier = train_config['decay_multiplier']
    decay_epochs = [int(epoch_num) for epoch_num in train_config['decay_epochs'].split(',')]
    lr_base = train_config['learning_rate']
    if lr_scheduler_name == 'multi_step':
        lr_scheduler = MultiStepLR(optimizer, milestones=decay_epochs, gamma=decay_multiplier, verbose=False)
    elif lr_scheduler_name == 'cosine_annealing':
        t_max = decay_epochs[0]
        lr_min = lr_base * decay_multiplier
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=lr_min, verbose=False)
    else:
        logger.error(f"No implementation is available for {lr_scheduler_name}!")
        logger.info("You can add the corresponding implementation to get_lr_scheduler in utils/lr_scheduler.py")
        logger.info("Or, you can use --lr-scheduler multi_step|cosine_annealing")
        exit()

    return lr_scheduler
