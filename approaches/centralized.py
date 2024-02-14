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

from utils.dataset import create_dataset
from utils.model import build_model
from utils.loss_function import get_loss_function
from utils.optimizer import get_optimizer
from utils.lr_scheduler import get_lr_scheduler

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Tuple


import logging
logger = logging.getLogger("centralized")


class CentralizedApproach:
    def __init__(self, dataset_config: dict, train_config: dict, model_config: dict,
                 loss_func_config: dict, optimizer_config: dict) -> None:

        # Dataset and DataLoader config
        num_workers = dataset_config['num_workers']
        batch_size = dataset_config['batch_size']
        train_data, test_data = create_dataset(dataset_config=dataset_config)
        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers)
        self.test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers)
        self.train_size = len(train_data)
        self.test_size = len(test_data)

        # Device config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model config
        self.model = build_model(model_config=model_config)
        self.model = self.model.to(self.device)

        # Loss function config
        self.loss_function_name = loss_func_config['loss_func_name']
        self.loss_function = get_loss_function(self.loss_function_name)

        # Optimizer config
        self.optimizer = get_optimizer(model_param=self.model.parameters(), optimizer_config=optimizer_config)

        # Train config
        if train_config['lr_scheduler_name']:
            self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer, train_config=train_config)

    def train_on_batch(self, image_batch: Tensor, label_batch: Tensor) -> Tuple[float, float]:
        image_batch = image_batch.to(self.device)
        label_batch = label_batch.to(self.device)
        self.model = self.model.to(self.device)

        self.model.train()

        output_batch = self.model(image_batch)
        loss = self.loss_function(output_batch, label_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        _, predicted_labels = output_batch.max(1)
        num_correct_predictions = predicted_labels.eq(label_batch).sum().item()

        train_loss = loss.item()
        train_accuracy = num_correct_predictions / label_batch.size(0)

        return train_loss, train_accuracy

    def train_model(self) -> Tuple[float, float]:
        num_correct_predictions = 0
        loss_total = 0.0
        for image_batch, label_batch in self.train_loader:
            batch_loss, batch_accuracy = self.train_on_batch(image_batch, label_batch)

            loss_total += batch_loss * label_batch.size(0)
            num_correct_predictions += batch_accuracy * label_batch.size(0)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

        train_loss = loss_total / self.train_size
        train_accuracy = num_correct_predictions / self.train_size

        return train_loss, train_accuracy

    def evaluate_model(self) -> Tuple[float, float]:

        self.model = self.model.to(self.device)
        self.model.eval()

        num_correct_predictions = 0
        loss_total = 0.0
        with torch.no_grad():
            for image_batch, label_batch in self.test_loader:
                image_batch = image_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                output_batch = self.model(image_batch)
                loss = self.loss_function(output_batch, label_batch)
                _, predicted_labels = output_batch.max(1)
                num_correct_predictions += predicted_labels.eq(label_batch).sum().item()
                loss_total += loss.item() * label_batch.size(0)

            test_loss = loss_total / self.test_size
            test_accuracy = num_correct_predictions / self.test_size

        return test_loss, test_accuracy

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def save_checkpoint(self, last_epoch, checkpoint_path):
        checkpoint_dict = {
            'last_epoch': last_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if hasattr(self, 'lr_scheduler'):
            checkpoint_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['last_epoch']

        if 'scheduler_state_dict' in checkpoint.keys():
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return last_epoch
