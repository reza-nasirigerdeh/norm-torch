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

import logging
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from approaches.base_approach import BaseApproach
from utils.dataset import Dataset
from utils.utils import evaluate

logger = logging.getLogger("centralized")


class CentralizedApproach(BaseApproach):
    def __init__(self, dataset_config, train_config, model_config, loss_func_config, optimizer_config):

        # Dataset configuration
        dataset_name = dataset_config['name']
        resize_shape_train = dataset_config['resize_shape_train']
        resize_shape_test = dataset_config['resize_shape_test']
        hflip = dataset_config['hflip']
        crop_shape = dataset_config['crop_shape']
        crop_padding = dataset_config['crop_padding']
        resized_crop_shape = dataset_config['resized_crop_shape']
        center_crop_shape = dataset_config['center_crop_shape']
        norm_mean = dataset_config['norm_mean']
        norm_std = dataset_config['norm_std']

        dataset = Dataset(dataset_name=dataset_name,
                          resize_shape_train=resize_shape_train,
                          resize_shape_test=resize_shape_test,
                          hflip=hflip,
                          crop_shape=crop_shape,
                          crop_padding=crop_padding,
                          resized_crop_shape=resized_crop_shape,
                          center_crop_shape=center_crop_shape,
                          norm_mean=norm_mean,
                          norm_std=norm_std)

        num_workers = train_config['num_workers']
        self.test_loader = torch.utils.data.DataLoader(dataset=dataset.test_set, batch_size=100, shuffle=False,
                                                       num_workers=num_workers)
        self.test_size = len(dataset.test_set)

        # extend model config
        model_config['num_classes'] = dataset.num_classes

        super(CentralizedApproach, self).__init__(train_dataset=dataset.train_set, train_config=train_config,
                                                  model_config=model_config, loss_func_config=loss_func_config,
                                                  optimizer_config=optimizer_config)

        # learning rate scheduler
        lr_scheduler = optimizer_config['lr_scheduler']
        if lr_scheduler:
            decay_epochs = [int(epoch_num) for epoch_num in optimizer_config['decay_epochs'].split(',')]
            decay_multiplier = optimizer_config['decay_multiplier']
            if lr_scheduler == 'multi_step':
                self.lr_scheduler = MultiStepLR(self.optimizer, milestones=decay_epochs, gamma=decay_multiplier, verbose=True)
            elif lr_scheduler == 'cosine_annealing':
                t_max = decay_epochs[0]
                lr_min = optimizer_config['learning_rate'] * decay_multiplier
                self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=lr_min, verbose=True)
            else:
                print(f"{lr_scheduler} is an invalid lr scheduler name")
                print("Exiting ....")
                exit()

    def train_model(self):
        num_correct_predictions = 0
        loss_total = 0.0
        for image_batch, label_batch in self.train_loader:
            batch_loss, batch_accuracy = self.train_on_batch(image_batch, label_batch)

            loss_total += batch_loss * label_batch.size(0)
            num_correct_predictions += batch_accuracy * label_batch.size(0)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

        train_loss = loss_total / len(self.train_dataset)
        train_accuracy = num_correct_predictions / len(self.train_dataset)

        return train_loss, train_accuracy

    def evaluate_model(self):
        return evaluate(self.model, self.test_loader, self.test_size, self.loss_function, self.device)
