#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_dataset
from noise2noise import Noise2Noise
from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Speckle2Speckle')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/valid')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=2, type=int)
    parser.add_argument('-ts', '--train-size', help='size of train dataset', type=int)
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', type=int)

    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function',
                        choices=['l1', 'l2', 'hdr',
                                 'l1+interface', 'l2+interface', 'hdr+interface',
                                 'l1+interfacePSF', 'l2+interfacePSF', 'hdr+interfacePSF',
                                 'interface_l1', 'interface_l2'],
                        default='l1', type=str)
    parser.add_argument('--interface_weight', help='interface loss weight', default=0.5, type=float)
    parser.add_argument('--interface_gauss_sigma', help='interface loss edge smoothing sigma', default=3, type=float)
    parser.add_argument('--interface_power', help='interface loss gradient magnitude power', default=0.8, type=float)
    parser.add_argument('--psf_gauss_sigma_x', help='PSF sigma X', default=2, type=float)
    parser.add_argument('--psf_gauss_sigma_y', help='PSF sigma Y', default=1, type=float)

    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
                        choices=['gaussian', 'poisson', 'text', 'mc', 'intrinsic'], default='intrinsic', type=str)
    parser.add_argument('-p', '--noise-param', help='noise parameter (e.g. std for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=128, type=int)
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')
    parser.add_argument('--average-validation-targets',
                        help='use averaging among targets for validation / testing', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    """Trains Speckle2Speckle."""

    # Parse training parameters
    params = parse_args()

    # Train/valid datasets
    train_loader = load_dataset(params.train_dir, params.train_size, params, shuffled=True)
    valid_loader = load_dataset(params.valid_dir, params.valid_size, params, shuffled=False,
                                target_averaging=params.average_validation_targets)

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)
    n2n.train(train_loader, valid_loader)
