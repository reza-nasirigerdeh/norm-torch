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

from approaches.centralized import CentralizedApproach
from utils.utils import ResultFile

import argparse

import logging
logger = logging.getLogger("simulate")


def main():

    # ############ Simulation parameters ##################################
    parser = argparse.ArgumentParser(description="Simulate mini-batch gradient descent training",
                                     usage=f"python simulate.py")

    parser.add_argument("--dataset_name", "--dataset", type=str, help="dataset name", default="cifar10")

    parser.add_argument("--dataset_dir", "--dataset-dir", type=str, help="root directory containing datasets", default="./datasets")

    parser.add_argument("--num_classes", "--num-classes", type=int, help="number of classes of the dataset", default=10)

    parser.add_argument("--resize_train", "--resize-train", type=str, help="shape for resizing train images e.g. 32x32", default="")

    parser.add_argument("--resize_test", "--resize-test", type=str, help="shape for resizing test images e.g. 32x32", default="")

    parser.add_argument("--random_crop", "--random-crop", type=str, help="random crop shape and padding e.g. 32x32-4x4", default="")

    parser.add_argument("--random_resized_crop", "--random-resized-crop", type=str, help="crop at a random location, and resize to e.g. 128x128", default="")

    parser.add_argument("--center_crop", "--center-crop", type=str, help="crop at the center with size e.g. 32x32", default="")

    parser.add_argument("--random_hflip", "--random-hflip", action='store_true', help="random horizontal flip with p=0.5", default=False)

    parser.add_argument("--norm_mean", "--norm-mean", type=str,
                        help="comma separated per-channel mean values of the dataset e.g. 0.5071,0.4867,0.4408", default="")

    parser.add_argument("--norm_std", "--norm-std", type=str,
                        help="comma separated per-channel std values of the dataset e.g. 0.2675,0.2565,0.2761", default="")

    parser.add_argument("--model_name", "--model", type=str, help="Model to be trained", default="cnn")

    parser.add_argument("--num_groups", "--num-groups", type=int, help="number of groups for GroupNorm", default=32)

    parser.add_argument("--kn_dropout_p", "--kn-dropout-p", type=float, help="dropout probability of KernelNorm", default=0.05)

    parser.add_argument("--low_resolution", "--low-resolution", action='store_true', help="3x3 conv instead of 7x7 conv as the first layer for low resolution images", default=False)

    parser.add_argument("--loss_function_name", "--loss", type=str, help="loss function name", default="cross_entropy")

    parser.add_argument("--optimizer_name", "--optimizer", type=str, help="optimizer name", default="sgd")

    parser.add_argument("--learning_rate", "--learning-rate", "--lr", type=float, help="learning rate", default=0.01)

    parser.add_argument("--lr_scheduler", "--lr-scheduler", type=str, help="learning rate scheduler e.g. multi_step or cosine_annealing", default="")

    parser.add_argument("--decay_epochs", "--decay-epochs", type=str, help="[comma separated list of] epoch in/over which learning rate decays should be applied", default='')

    parser.add_argument("--decay_multiplier", "--decay-multiplier", type=float, help="learning rate decay multiplier ", default=0.1)

    parser.add_argument("--momentum", "--momentum", type=float, help="momentum of SGD optimizer", default=0.0)

    parser.add_argument("--weight_decay", "--weight-decay", type=float, help="weight decay value", default=0.0)

    parser.add_argument("--dampening", "--dampening", type=float, help="dampening value for SGD optimizer", default=0.0)

    parser.add_argument("--nesterov", "--nesterov", action='store_true', help="nesterov flag for SGD optimizer", default=False)

    parser.add_argument("--batch_size", "--batch-size", type=int, help="batch size", default=128)

    parser.add_argument("--num_epochs", "--epochs", type=int, help="number of epochs", default=100)

    parser.add_argument("--num_workers", "--workers", type=int, help="number of workers", default=4)

    parser.add_argument("--log_level", "--log-level", type=str, help="log level", default='info')

    parser.add_argument("--run_number", "--run", type=int,
                        help="the run number to have a separate result file for each run ", default=1)

    parser.add_argument("--eval_freq", "--eval-freq", type=int, help="evaluate model every eval-freq epochs", default=1)

    parser.add_argument("--checkpoint_freq", "--checkpoint-freq", type=int, help="save training status every checkpoint-freq epochs", default=0)

    parser.add_argument("--load_from_checkpoint", "--load-from-checkpoint", action='store_true', help="load from the last checkpoint", default=False)

    args = parser.parse_args()

    # ############ Simulator configuration ##################################

    # ##### Dataset config
    shape_str = args.resize_train.split('x')
    resize_shape_train = (int(shape_str[0]), int(shape_str[1])) if len(shape_str) == 2 else ()

    shape_str = args.resize_test.split('x')
    if len(shape_str) == 2:
        resize_shape_test = (int(shape_str[0]), int(shape_str[1]))
    elif len(shape_str) == 1:
        resize_shape_test = int(shape_str[0]) if shape_str[0] else ()
    else:
        resize_shape_test = ()

    if "-" in args.random_crop:
        crop_shape_str = args.random_crop.split('-')[0].split('x')
        crop_shape = (int(crop_shape_str[0]), int(crop_shape_str[1])) if len(crop_shape_str) == 2 else ()

        crop_padding_str = args.random_crop.split('-')[1].split('x')
        crop_padding = (int(crop_padding_str[0]), int(crop_padding_str[1])) if len(crop_padding_str) == 2 else ()
    elif "x" in args.random_crop:
        crop_shape_str = args.random_crop.split('x')
        crop_shape = (int(crop_shape_str[0]), int(crop_shape_str[1])) if len(crop_shape_str) == 2 else ()

        crop_padding = ()
    else:
        crop_shape = ()
        crop_padding = ()

    resized_crop_shape_str = args.random_resized_crop.split('x')
    resized_crop_shape = (int(resized_crop_shape_str[0]), int(resized_crop_shape_str[1])) if len(resized_crop_shape_str) == 2 else ()

    center_crop_shape_str = args.center_crop.split('x')
    center_crop_shape = (int(center_crop_shape_str[0]), int(center_crop_shape_str[1])) if len(center_crop_shape_str) == 2 else ()

    norm_mean_str = args.norm_mean.split(',')
    norm_mean = (float(norm_mean_str[0]), float(norm_mean_str[1]), float(norm_mean_str[2])) if len(norm_mean_str) == 3 else ()

    norm_std_str = args.norm_std.split(',')
    norm_std = (float(norm_std_str[0]), float(norm_std_str[1]), float(norm_std_str[2])) if len(norm_std_str) == 3 else ()

    dataset_config = {'dataset_name': args.dataset_name, 'dataset_dir': args.dataset_dir,
                      'num_workers': args.num_workers, 'batch_size': args.batch_size,
                      'resize_shape_train': resize_shape_train, 'resize_shape_test': resize_shape_test,
                      'horiz_flip': args.random_hflip, 'crop_shape': crop_shape, 'crop_padding': crop_padding,
                      'resized_crop_shape': resized_crop_shape, 'center_crop_shape': center_crop_shape,
                      'norm_mean': norm_mean, 'norm_std': norm_std, 'run_number': args.run_number}

    # ##### Model config
    model_config = {'model_name': args.model_name, 'num_classes': args.num_classes, 'num_groups': args.num_groups,
                    'kn_dropout_p': args.kn_dropout_p, 'low_resolution': args.low_resolution}

    # ##### Loss function config
    loss_func_config = {'loss_func_name': args.loss_function_name}

    # ##### Optimizer config
    optimizer_config = {'optimizer_name': args.optimizer_name, 'learning_rate': args.learning_rate,
                        'momentum': args.momentum, 'weight_decay': args.weight_decay,
                        'dampening': args.dampening, 'nesterov': args.nesterov,
                        }

    train_config = {'lr_scheduler_name': args.lr_scheduler, 'learning_rate': args.learning_rate,
                    'decay_epochs': args.decay_epochs, 'decay_multiplier': args.decay_multiplier
                    }

    # ##### Logger config
    logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=args.log_level.upper())

    # ##### Approach instantiation
    cent_approach = CentralizedApproach(dataset_config=dataset_config, train_config=train_config,
                                        model_config=model_config, loss_func_config=loss_func_config,
                                        optimizer_config=optimizer_config)

    # ##### Result file config
    aug_str = ""
    aug_str += 'r' if resize_shape_train else ""
    aug_str += 'f' if args.random_hflip else ""
    aug_str += 'c' if (crop_shape or resized_crop_shape) else ""
    aug_str += 'n' if norm_mean and norm_std else ""
    if aug_str:
        aug_str = '_' + aug_str
    dataset_str = args.dataset_name + f'{aug_str}'

    model_str = args.model_name + (f'-G{args.num_groups}' if 'gn' in args.model_name else "")

    optimizer_str = args.optimizer_name + (f'-m{args.momentum}' if (args.optimizer_name == 'sgd') else "") + \
        f'-wd{args.weight_decay}' + f'-dmp{args.dampening}' + ('-nesterov' if args.nesterov else "") + (f"-{args.lr_scheduler}" if args.lr_scheduler else "") + \
        f"-lr{args.learning_rate}"

    loss_func_str = args.loss_function_name

    train_str = f"bs{args.batch_size}"

    result_file_name = f"{dataset_str}-{model_str}-{loss_func_str}-" \
                       f"{optimizer_str}-{train_str}"
    result_file_name += f"-run{args.run_number}"

    checkpoint_path = f'./results/{result_file_name}.pt'

    result_file_name += ".csv"

    mode = 'a' if args.load_from_checkpoint else 'w'
    result_file = ResultFile(result_file_name=result_file_name, mode=mode)
    if not args.load_from_checkpoint:
        result_file.write_header('epoch,train_loss,test_loss,train_accuracy,test_accuracy')

    # ############ Print simulation parameters ##################################
    logger.info(f"\n-------------- SIMULATION PARAMETERS ----------------")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Dataset directory: {args.dataset_dir}")
    if resize_shape_train:
        logger.info(f'Resize shape of train images: {resize_shape_train}')
    if crop_shape:
        logger.info(f'Random crop shape: {crop_shape}')
    if crop_padding:
        logger.info(f'Random crop padding: {crop_padding}')
    if resized_crop_shape:
        logger.info(f'Random resized crop shape: {resized_crop_shape}')
    if resize_shape_test:
        logger.info(f'Resize shape of test images: {resize_shape_test}')
    if center_crop_shape:
        logger.info(f'Center crop shape of test images: {center_crop_shape}')
    if args.random_hflip:
        logger.info(f'Random horizontal flip: {args.random_hflip}')
    if args.norm_mean:
        logger.info(f"Norm mean: {norm_mean}")
    if args.norm_std:
        logger.info(f"Norm std: {norm_std}")

    logger.info(f"Model: {args.model_name}")
    if 'gn' in args.model_name:
        logger.info(f"Number of Groups: {args.num_groups}")

    logger.info(f"Optimizer: {args.optimizer_name}")
    logger.info(f"Loss function: {args.loss_function_name}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    if args.optimizer_name == 'sgd':
        logger.info(f"Momentum: {args.momentum}")
        logger.info(f"Dampening: {args.dampening}")
        logger.info(f"Nesterov: {args.nesterov}")

    logger.info(f"Weight decay: {args.weight_decay}")

    if args.lr_scheduler:
        logger.info(f"LR scheduler: {args.lr_scheduler}")
        logger.info(f"Decay epochs: {args.decay_epochs}")
        logger.info(f"Decay multiplier: {args.decay_multiplier}")

    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Evaluation freq: {args.eval_freq}")

    if args.checkpoint_freq == 0:
        logger.info(f"Save checkpoint: False")
    else:
        logger.info(f"Save checkpoint: True, every {args.checkpoint_freq} epochs")

    if args.load_from_checkpoint:
        logger.info(f"Resume from last checkpoint: {args.load_from_checkpoint}")

    logger.info(f"Run number: {args.run_number}")

    # load checkpoint if its flag is set
    if args.load_from_checkpoint:
        logger.info("Loading from the last checkpoint ....")
        last_epoch = cent_approach.load_checkpoint(checkpoint_path)
    else:
        last_epoch = 0

    # ############ Run simulation ##################################
    logger.info(f"\n-------------- SIMULATION STARTED ----------------")

    last_epoch += 1

    for current_epoch in range(last_epoch, args.num_epochs + 1):
        logger.info(f"--- Epoch # {current_epoch}")

        # train model
        logger.info(f'Adjusting learning rate to {cent_approach.get_lr():e}.')

        train_loss, train_accuracy = cent_approach.train_model()
        logger.info(f'Train loss: {train_loss:.6f}, Train accuracy: {train_accuracy:.6f}')

        if current_epoch % args.eval_freq == 0 or current_epoch == 1 or current_epoch > (args.num_epochs - 5):
            # test model
            test_loss, test_accuracy = cent_approach.evaluate_model()
            logger.info(f'Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.6f}')

            # write result
            result_file.write_result(epoch=current_epoch,
                                     result_list=[train_loss, test_loss, train_accuracy, test_accuracy])

        if args.checkpoint_freq != 0 and current_epoch % args.checkpoint_freq == 0:
            logger.info("Saving checkpoint ....")
            cent_approach.save_checkpoint(last_epoch=current_epoch, checkpoint_path=checkpoint_path)

    result_file.close()
    logger.info("Saving checkpoint ....")
    cent_approach.save_checkpoint(last_epoch=args.num_epochs, checkpoint_path=checkpoint_path)
    logger.info(f"-------------- SIMULATION DONE ----------------")


if __name__ == "__main__":
    main()
