#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.optim import Adam, lr_scheduler

import kornia.filters

import math
import numbers

from unet import UNet
from utils import *

import os
import json


class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self._compile()


    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        # Model (3x3=9 channels for Monte Carlo since it uses 3 HDR buffers)
        if self.p.noise_type == 'intrinsic':
            self.is_mc = False
            self.model = UNet(in_channels=1, out_channels=1)
        elif self.p.noise_type == 'mc':
            self.is_mc = True
            self.model = UNet(in_channels=9)
        else:
            self.is_mc = False
            self.model = UNet(in_channels=3)

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            # Learning rate adjustment
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                patience=self.p.nb_epochs/4, factor=0.5, verbose=True)

            # Loss function
            if self.p.loss.startswith('interface'):
                if self.p.loss == 'interface_l2':
                    self.loss = nn.MSELoss()
                elif self.p.loss == 'interface_l1':
                    self.loss = nn.L1Loss()
                self.loss = InterfaceMapLoss(
                    self.loss,
                    self.p.interface_gauss_sigma)
                self.interface_loss = True
            else:
                if self.p.loss.startswith('hdr'):
                    assert self.is_mc, 'Using HDR loss on non Monte Carlo images'
                    self.loss = HDRLoss()
                elif self.p.loss.startswith('l2'):
                    self.loss = nn.MSELoss()
                else:
                    self.loss = nn.L1Loss()

                if self.p.loss.endswith('+interface'):
                    self.loss = InterfaceLoss(
                        self.loss,
                        self.p.interface_weight,
                        self.p.interface_gauss_sigma,
                        self.p.interface_power)
                    self.interface_loss = True
                elif self.p.loss.endswith('+interfacePSF'):
                    self.loss = InterfacePSFLoss(
                        self.loss,
                        self.p.interface_weight,
                        self.p.interface_gauss_sigma,
                        self.p.interface_power,
                        (self.p.psf_gauss_sigma_y, self.p.psf_gauss_sigma_x))
                    self.interface_loss = True

                else:
                    self.interface_loss = False
        else:
            self.model.train(False)
            for param in self.model.parameters():
                param.requires_grad = False

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()


    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()


    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            if self.p.clean_targets:
                ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-clean-%H%M}'
            else:
                ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-%H%M}'
            if self.p.ckpt_overwrite:
                if self.p.clean_targets:
                    ckpt_dir_name = f'{self.p.noise_type}-clean'
                else:
                    ckpt_dir_name = self.p.noise_type

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/n2n.pt'.format(self.ckpt_dir)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/n2n-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/n2n-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)

    def save_script_module(self, epoch, stats, example, first=False):
        # Create directory for model checkpoints, if nonexistent
        if first:
            if self.p.clean_targets:
                ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-clean-%H%M}'
            else:
                ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-%H%M}'
            if self.p.ckpt_overwrite:
                if self.p.clean_targets:
                    ckpt_dir_name = f'{self.p.noise_type}-clean'
                else:
                    ckpt_dir_name = self.p.noise_type

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/n2n-jit.pt'.format(self.ckpt_dir)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/n2n-jit-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)

        traced_script_module = torch.jit.trace(self.model, example)
        print('Saving script module to: {}\n'.format(fname_unet))
        traced_script_module.save(fname_unet)

    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))


    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader, example=None):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        self.save_model(epoch, stats, epoch == 0)
        if example is not None:
            self.save_script_module(epoch, stats, example, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            loss_str = f'{self.p.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')


    def test(self, test_loader, show):
        """Evaluates denoiser on test set."""

        self.model.train(False)

        source_imgs = []
        denoised_imgs = []
        clean_imgs = []

        # Create directory for denoised images
        save_path = os.path.join(self.p.output_path, 'denoised')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for batch_idx, (source, target) in enumerate(test_loader):
            # Only do first <show> images
            if show == 0 or batch_idx >= show:
                break

            if isinstance(target, dict):
                target_interfaces = target['interfaces']
                target = target['image']
            else:
                target_interfaces = None

            source_imgs.append(source)
            clean_imgs.append(target)

            if self.use_cuda:
                source = source.cuda()

            # Denoise
            denoised_img = self.model(source).detach()
            denoised_imgs.append(denoised_img)

        # Squeeze tensors
        source_imgs = [t.squeeze(0) for t in source_imgs]
        denoised_imgs = [t.squeeze(0) for t in denoised_imgs]
        clean_imgs = [t.squeeze(0) for t in clean_imgs]

        # Create montage and save images
        print('Saving images and montages to: {}'.format(save_path))
        for i in range(len(source_imgs)):
            if len(test_loader.dataset.imgs) > i:
                img_name = test_loader.dataset.imgs[i]
            else:
                img_name = str(i)
            create_montage(img_name, self.p.noise_type, save_path, source_imgs[i], denoised_imgs[i], clean_imgs[i], show)

    def apply(self, images, transpose=True):
        self.model.train(False)

        # determine required padding
        required_multiples = 32
        if transpose:
            image_width = images.shape[2]
            image_height = images.shape[1]
        else:
            image_width = images.shape[1]
            image_height = images.shape[2]
        padding_total = [
            math.ceil(image_width / required_multiples) * required_multiples - image_width,
            math.ceil(image_height / required_multiples) * required_multiples - image_height]
        padding = [
            math.ceil(padding_total[1] / 2), math.floor(padding_total[1] / 2),
            math.ceil(padding_total[0] / 2), math.floor(padding_total[0] / 2)]

        # create output image
        output = images.copy() * 0
        for image_index in range(images.shape[0]):
            if transpose:
                source = torch.from_numpy(images[image_index, :, :].T)
            else:
                source = torch.from_numpy(images[image_index, :, :])
            source = torch.unsqueeze(torch.unsqueeze(source, 0), 0)
            if self.use_cuda:
                source = source.cuda()

            # Denoise
            denoised_img = self.model(nnf.pad(source, padding, mode='replicate')).detach()
            # crop
            denoised_img = denoised_img.squeeze()
            denoised_img = denoised_img[padding[2]:-padding[3], padding[0]:-padding[1]]

            # store output
            if transpose:
                output[image_index, :, :] = denoised_img.cpu().numpy().T
            else:
                output[image_index, :, :] = denoised_img.cpu().numpy()

        return output

    def eval(self, valid_loader):
        """Evaluates denoiser on validation set."""

        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()

        for batch_idx, (source, target) in enumerate(valid_loader):
            if isinstance(target, dict):
                target_interfaces = target['interfaces']
                target = target['image']
            else:
                target_interfaces = None

            if self.use_cuda:
                source = source.cuda()
                target = target.cuda()
                if target_interfaces is not None:
                    target_interfaces = target_interfaces.cuda()

            # Denoise
            source_denoised = self.model(source)

            # Update loss
            if self.interface_loss:
                loss = self.loss(source_denoised, target, target_interfaces)
            else:
                loss = self.loss(source_denoised, target)
            loss_meter.update(loss.item())

            # Compute PSRN
            if self.is_mc:
                source_denoised = reinhard_tonemap(source_denoised)
            # TODO: Find a way to offload to GPU, and deal with uneven batch sizes
            for i in range(self.p.batch_size):
                source_denoised = source_denoised.cpu()
                target = target.cpu()
                psnr_meter.update(psnr(source_denoised[i], target[i]).item())

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg

        return valid_loss, valid_time, psnr_avg


    def train(self, train_loader, valid_loader):
        """Trains denoiser on training set."""

        self.model.train(True)

        self._print_params()
        num_batches = len(train_loader)
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'

        # Dictionaries of tracked stats
        stats = {'noise_type': self.p.noise_type,
                 'noise_param': self.p.noise_param,
                 'train_loss': [],
                 'valid_loss': [],
                 'valid_psnr': []}

        # Main training loop
        train_start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, (source, target) in enumerate(train_loader):
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                if isinstance(target, dict):
                    target_interfaces = target['interfaces']
                    target = target['image']
                else:
                    target_interfaces = None

                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()
                    if target_interfaces is not None:
                        target_interfaces = target_interfaces.cuda()

                # Denoise image
                source_denoised = self.model(source)

                if self.interface_loss:
                    loss = self.loss(source_denoised, target, target_interfaces)
                else:
                    loss = self.loss(source_denoised, target)
                loss_meter.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

            # Epoch end, save and reset tracker
            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader, example=source)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))


class HDRLoss(nn.Module):
    """High dynamic range loss."""

    def __init__(self, eps=0.01):
        """Initializes loss with numerical stability epsilon."""

        super(HDRLoss, self).__init__()
        self._eps = eps

    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""

        loss = ((denoised - target) ** 2) / (denoised + self._eps) ** 2
        return torch.mean(loss.view(-1))


class InterfaceLoss(nn.Module):
    """This loss function combines a data loss with interface sharpness."""

    def __init__(self, data_loss, interface_weight, interface_gauss_sigma, interface_power):
        super(InterfaceLoss, self).__init__()

        kernel_size = int(math.floor(interface_gauss_sigma*3.0)) * 2 + 1
        self._gauss = kornia.filters.GaussianBlur2d(
            (kernel_size, kernel_size), (interface_gauss_sigma, interface_gauss_sigma))
        self._grad = kornia.filters.Sobel()
        self._data_loss = data_loss
        self._interface_weight = interface_weight
        self._interface_power = interface_power

    def forward(self, denoised, target, target_interfaces):
        edge_weights = self._gauss(target_interfaces)
        # interface_edges = torch.pow(self._grad(denoised) * edge_weights, self._interface_power)
        # interface_loss = torch.pow(torch.mean(interface_edges.view(-1)), 1/self._interface_power)

        interface_edgesA = self._grad(denoised)
        interface_edgesB = interface_edgesA * edge_weights
        interface_edges = torch.pow(torch.abs(interface_edgesB) + 1e-7, self._interface_power)

        interface_lossA = torch.mean(interface_edges.view(-1))
        interface_loss = torch.pow(interface_lossA, 1.0 / self._interface_power)

        #interface_edgesA.retain_grad()
        #interface_edgesB.retain_grad()
        #interface_edges.retain_grad()
        #interface_lossA.retain_grad()
        #interface_loss.retain_grad()

        data_loss = self._data_loss(denoised, target)
        return data_loss + self._interface_weight * interface_loss


class InterfacePSFLoss(nn.Module):
    """This loss function combines a data loss with interface sharpness."""

    def __init__(self, data_loss, interface_weight, interface_gauss_sigma, interface_power, psf_sigma):
        super(InterfacePSFLoss, self).__init__()

        kernel_size = int(math.floor(interface_gauss_sigma*3.0)) * 2 + 1
        self._gauss = kornia.filters.GaussianBlur2d(
            (kernel_size, kernel_size), (interface_gauss_sigma, interface_gauss_sigma))
        kernel_size = (
                int(math.floor(psf_sigma[0] * 2.0)) * 2 + 1,
                int(math.floor(psf_sigma[1] * 2.0)) * 2 + 1)
        self._gauss_psf = kornia.filters.GaussianBlur2d(
            kernel_size, psf_sigma)
        # FOR REPLACING THE GAUSSIAN FILTER WITH A HANN WINDOW
        # kernel_size = (
        #         int(math.floor(psf_sigma[0] * 2.0)) * 2 + 1,
        #         int(math.floor(psf_sigma[1] * 2.0)) * 2 + 1)
        # self._gauss_psf = kornia.filters.GaussianBlur2d(
        #     kernel_size, psf_sigma)
        # hann_axial = torch.hann_window(kernel_size[0]+2, periodic=False)
        # hann_lateral = torch.hann_window(kernel_size[1] + 2, periodic=False)
        # hann_axial = hann_axial[1:-1]
        # hann_lateral = hann_lateral[1:-1]
        # self._gauss_psf.kernel = (hann_axial.reshape((kernel_size[0], 1)) * hann_lateral.reshape((1, kernel_size[1]))).reshape(self._gauss_psf.kernel.shape)
        # self._gauss_psf.kernel /= torch.sum(self._gauss_psf.kernel)
        self._grad = kornia.filters.Sobel()
        self._data_loss = data_loss
        self._interface_weight = interface_weight
        self._interface_power = interface_power
        self._mse = torch.nn.MSELoss()

    def forward(self, denoised, target, target_interfaces):
        edge_weights = self._gauss(target_interfaces)
        edge_weights_min = edge_weights.min()
        edge_weights_max = edge_weights.max() + 1e-7
        edge_weights = (edge_weights - edge_weights_min) / (edge_weights_max - edge_weights_min)
        # interface_edges = torch.pow(self._grad(denoised) * edge_weights, self._interface_power)
        # interface_loss = torch.pow(torch.mean(interface_edges.view(-1)), 1/self._interface_power)

        denoised_smoothed = self._gauss_psf(denoised)
        interface_loss = self._mse(denoised_smoothed * edge_weights, target * edge_weights)

        data_weights = 1 - edge_weights
        data_loss = self._data_loss(denoised * data_weights, target * data_weights)
        return data_loss + self._interface_weight * interface_loss


class InterfaceMapLoss(nn.Module):
    """This loss function uses a given data loss against the interface indicator map."""

    def __init__(self, data_loss, interface_gauss_sigma):
        super(InterfaceMapLoss, self).__init__()

        kernel_size = int(math.floor(interface_gauss_sigma*3.0)) * 2 + 1
        self._gauss = kornia.filters.GaussianBlur2d(
            (kernel_size, kernel_size), (interface_gauss_sigma, interface_gauss_sigma))
        self._grad = kornia.filters.Sobel()
        self._data_loss = data_loss

    def forward(self, denoised, target, target_interfaces):
        interfaces_smoothed = self._gauss(target_interfaces)

        data_loss = self._data_loss(denoised, interfaces_smoothed)
        return data_loss