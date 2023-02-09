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

import wget
import tarfile
import os
import logging

import numpy as np
from torchvision import datasets, transforms

logger = logging.getLogger("dataset")

dataset_root = 'datasets'


class Dataset:
    def __init__(self, dataset_name, resize_shape_train=(), resize_shape_test=(), hflip=False,
                 crop_shape=(), crop_padding=(), resized_crop_shape=(), center_crop_shape=(),
                 norm_mean=(), norm_std=()):

        self.name = dataset_name

        logger.info(f"Loading dataset {self.name} ...")

        # create dataset_root folder, if it does not already exist
        if not os.path.exists(dataset_root):
            os.mkdir(dataset_root)

        # local function to get the train/test transform object
        def get_transform(train=True):
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

                if hflip:
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

        if self.name == 'mnist':
            train_data = datasets.MNIST(root=dataset_root, download=True, train=True, transform=train_transform)
            test_data = datasets.MNIST(root=dataset_root, download=False, train=False, transform=test_transform)

        elif self.name == 'fashion_mnist':
            train_data = datasets.FashionMNIST(root=dataset_root, download=True, train=True, transform=train_transform)
            test_data = datasets.FashionMNIST(root=dataset_root, download=False, train=False, transform=test_transform)

        elif self.name == 'cifar10':
            train_data = datasets.CIFAR10(root=dataset_root, download=True, train=True, transform=train_transform)
            test_data = datasets.CIFAR10(root=dataset_root, download=False, train=False, transform=test_transform)

        elif self.name == 'cifar100':
            train_data = datasets.CIFAR100(root=dataset_root, download=True, train=True, transform=train_transform)
            test_data = datasets.CIFAR100(root=dataset_root, download=False, train=False, transform=test_transform)

        elif self.name == 'imagenette_160px':
            # if imagenette-160px has not already been downloaded
            if not os.path.exists(f'{dataset_root}/imagenette2-160'):
                # download imagenette-160px dataset
                logger.info("Downloading the dataset ...")
                file_path = wget.download(url='https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',
                                          out=f'{dataset_root}')

                # extract tgz file
                logger.info("Extracting the dataset ...")
                tar = tarfile.open(name=file_path, mode="r:gz")
                tar.extractall(path=f'{dataset_root}')
                tar.close()

            train_data = datasets.ImageFolder(root=f'{dataset_root}/imagenette2-160/train', transform=train_transform)
            test_data = datasets.ImageFolder(root=f'{dataset_root}/imagenette2-160/val', transform=test_transform)

        elif self.name == 'imagenette_320px':
            if not os.path.exists(f'{dataset_root}/imagenette2-320'):
                # download imagenette-320px dataset
                logger.info("Downloading the dataset ...")
                file_path = wget.download(url='https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz',
                                          out=f'{dataset_root}')

                # extract tgz file
                logger.info("Extracting the dataset ...")
                tar = tarfile.open(name=file_path, mode="r:gz")
                tar.extractall(path=f'{dataset_root}')
                tar.close()

            train_data = datasets.ImageFolder(root=f'{dataset_root}/imagenette2-320/train', transform=train_transform)
            test_data = datasets.ImageFolder(root=f'{dataset_root}/imagenette2-320/val', transform=test_transform)

        elif self.name == 'imagewoof_160px':
            # if imagewoof-160px has not already been downloaded
            if not os.path.exists(f'{dataset_root}/imagewoof2-160'):
                # download imagenette-160px dataset
                logger.info("Downloading the dataset ...")
                file_path = wget.download(url='https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz',
                                          out=f'{dataset_root}')

                # extract tgz file
                logger.info("Extracting the dataset ...")
                tar = tarfile.open(name=file_path, mode="r:gz")
                tar.extractall(path=f'{dataset_root}')
                tar.close()

            train_data = datasets.ImageFolder(root=f'{dataset_root}/imagewoof2-160/train', transform=train_transform)
            test_data = datasets.ImageFolder(root=f'{dataset_root}/imagewoof2-160/val', transform=test_transform)

        elif self.name == 'imagewoof_320px':
            # if imagewoof-320px has not already been downloaded
            if not os.path.exists(f'{dataset_root}/imagewoof2-320'):
                # download imagenette-320px dataset
                logger.info("Downloading the dataset ...")
                file_path = wget.download(url='https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz',
                                          out=f'{dataset_root}')

                # extract tgz file
                logger.info("Extracting the dataset ...")
                tar = tarfile.open(name=file_path, mode="r:gz")
                tar.extractall(path=f'{dataset_root}')
                tar.close()

            train_data = datasets.ImageFolder(root=f'{dataset_root}/imagewoof2-320/train', transform=train_transform)
            test_data = datasets.ImageFolder(root=f'{dataset_root}/imagewoof2-320/val', transform=test_transform)

        elif self.name == 'imagenet':
            train_data = datasets.ImageFolder(root=f'{dataset_root}/imagenet/train', transform=train_transform)
            test_data = datasets.ImageFolder(root=f'{dataset_root}/imagenet/val', transform=test_transform)

        else:
            logger.error(f"No implementation is available for {dataset_name}!")
            logger.error("You can add the corresponding implementation to load_image_dataset in utils/dataset.py")
            logger.error("Or, you can use --dataset mnist|fashion_mnist|cifar10|cifar100|imagenette_160px|imagenette_320px")
            exit()

        self.train_set = train_data
        self.test_set = test_data
        self.num_classes = len(np.unique([train_data[index][1] for index in range(len(train_data))]))
