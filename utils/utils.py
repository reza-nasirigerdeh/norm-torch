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

import os
import torch
import numpy as np

import logging
logger = logging.getLogger("utils")


def evaluate(model, test_loader, test_size, loss_function, device):

    # initialize test data loader
    model = model.to(device)
    model.eval()

    num_correct_predictions = 0
    loss_total = 0.0
    with torch.no_grad():
        for image_batch, label_batch in test_loader:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            output_batch = model(image_batch)
            loss = loss_function(output_batch, label_batch)
            _, predicted_labels = output_batch.max(1)
            num_correct_predictions += predicted_labels.eq(label_batch).sum().item()
            loss_total += loss.item() * label_batch.size(0)

        test_loss = loss_total / test_size
        test_accuracy = num_correct_predictions / test_size

    return test_loss, test_accuracy


class ResultFile:
    def __init__(self, result_file_name):
        # create root directory for results
        result_root = './results'
        if not os.path.exists(result_root):
            os.mkdir(result_root)

        # open result file
        self.result_file = open(file=f'{result_root}/{result_file_name}', mode='w')

    def write_header(self, header):
        self.result_file.write(f'{header}\n')
        self.result_file.flush()

    def write_result(self, epoch, result_list):
        digits_precision = 8

        result_str = f'{epoch},'
        for result in result_list:
            if result != '-':
                result = np.round(result, digits_precision)
            result_str += f'{result},'

        # remove final comma
        result_str = result_str[0:-1]

        self.result_file.write(f'{result_str}\n')
        self.result_file.flush()

    def close(self):
        self.result_file.close()
