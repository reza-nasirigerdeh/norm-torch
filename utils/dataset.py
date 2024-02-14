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

import torch
import os
from torchvision import datasets, transforms
from typing import Tuple

import logging
logger = logging.getLogger("dataset")


def create_dataset(dataset_config: dict) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

    dataset_name = dataset_config['dataset_name']
    dataset_dir = dataset_config['dataset_dir']
    resize_shape_train = dataset_config['resize_shape_train']
    resize_shape_test = dataset_config['resize_shape_test']
    horiz_flip = dataset_config['horiz_flip']
    crop_shape = dataset_config['crop_shape']
    crop_padding = dataset_config['crop_padding']
    resized_crop_shape = dataset_config['resized_crop_shape']
    center_crop_shape = dataset_config['center_crop_shape']
    norm_mean = dataset_config['norm_mean']
    norm_std = dataset_config['norm_std']

    logger.info(f"Loading dataset {dataset_name} .....")

    # create dataset_root folder, if it does not already exist
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # local function to get the train/test transform object
    def get_transform(train: bool) -> transforms.transforms.Compose:
        trans = []

        # random horizontal flip, random_cropping, and random_resized_cropping is only applied to train images
        if train:
            if resize_shape_train:
                trans.append(transforms.Resize(size=resize_shape_train))

            if resized_crop_shape:
                trans.append(transforms.RandomResizedCrop(size=resized_crop_shape))

            if crop_shape:
                padding = crop_padding if crop_padding else None
                trans.append(transforms.RandomCrop(size=crop_shape, padding=padding))

            if horiz_flip:
                trans.append(transforms.RandomHorizontalFlip())
        else:
            if resize_shape_test:
                trans.append(transforms.Resize(size=resize_shape_test))

            if center_crop_shape:
                trans.append(transforms.CenterCrop(size=center_crop_shape))

        trans.append(transforms.ToTensor())

        if norm_mean and norm_std:
            trans.append(transforms.Normalize(mean=norm_mean, std=norm_std))

        return transforms.Compose(trans)

    train_transform = get_transform(train=True)
    test_transform = get_transform(train=False)

    if dataset_name == 'cifar10':
        train_data = datasets.CIFAR10(root=dataset_dir, download=True, train=True, transform=train_transform)
        test_data = datasets.CIFAR10(root=dataset_dir, download=False, train=False, transform=test_transform)

    elif dataset_name == 'cifar100':
        train_data = datasets.CIFAR100(root=dataset_dir, download=True, train=True, transform=train_transform)
        test_data = datasets.CIFAR100(root=dataset_dir, download=False, train=False, transform=test_transform)

    elif dataset_name == 'imagenet':
        train_data = datasets.ImageFolder(root=f'{dataset_dir}/imagenet/train', transform=train_transform)
        test_data = datasets.ImageFolder(root=f'{dataset_dir}/imagenet/val', transform=test_transform)
    else:
        logger.error(f"No implementation is available for {dataset_name}!")
        logger.error("You can add the corresponding implementation to create_dataset in utils/dataset.py")
        logger.error("Or, you can use --dataset cifar10|cifar100|imagenet")
        exit()

    return train_data, test_data
