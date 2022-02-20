#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision.transforms.functional as tvf
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted

from utils import load_hdr_as_tensor

import os
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw

import glob
import os.path
import re
from operator import itemgetter
from itertools import groupby
import psutil

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')


def load_dataset(root_dir, redux, params, shuffled=False, single=False, target_averaging=False, random_order=False):
    """Loads dataset and returns corresponding data loader."""

    # Create Torch dataset
    noise = (params.noise_type, params.noise_param)

    # Instantiate appropriate dataset class
    if params.noise_type == 'intrinsic':
        dataset = IntrinsicNoisyDataset(root_dir, redux, params.crop_size,
                                        seed=params.seed, target_averaging=target_averaging, random_order=random_order)
    elif params.noise_type == 'mc':
        dataset = MonteCarloDataset(root_dir, redux, params.crop_size,
            clean_targets=params.clean_targets)
    else:
        dataset = NoisyDataset(root_dir, redux, params.crop_size,
            clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed)

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled,
                          num_workers=min(psutil.cpu_count(), 9))


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.clean_targets = clean_targets

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvf.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvf.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)


class NoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, redux, crop_size, clean_targets=False,
        noise_dist=('gaussian', 50.), seed=None):
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        self.imgs = os.listdir(root_dir)
        if redux:
            self.imgs = self.imgs[:redux]

        # Noise parameters (max std for Gaussian, lambda for Poisson, nb of artifacts for text)
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)


    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""

        w, h = img.size
        c = len(img.getbands())

        # Poisson distribution
        # It is unclear how the paper handles this. Poisson noise is not additive,
        # it is data dependent, meaning that adding sampled valued from a Poisson
        # will change the image intensity...
        if self.noise_type == 'poisson':
            noise = np.random.poisson(img)
            noise_img = img + noise
            noise_img = 255 * (noise_img / np.amax(noise_img))

        # Normal distribution (default)
        else:
            if self.seed:
                std = self.noise_param
            else:
                std = np.random.uniform(0, self.noise_param)
            noise = np.random.normal(0, std, (h, w, c))

            # Add noise and clip
            noise_img = np.array(img) + noise

        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_img)


    def _add_text_overlay(self, img):
        """Adds text overlay to images."""

        assert self.noise_param < 1, 'Text parameter is an occupancy probability'

        c = len(img.getbands())

        # Choose font and get ready to draw
        if platform == 'linux':
            serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            serif = 'Times New Roman.ttf'
        text_img = img.copy()
        text_draw = ImageDraw.Draw(text_img)

        # Text binary mask to compute occupancy efficiently
        w, h = img.size
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        # Random occupancy in range [0, p]
        if self.seed:
            random.seed(self.seed)
            max_occupancy = self.noise_param
        else:
            max_occupancy = np.random.uniform(0, self.noise_param)
        def get_occupancy(x):
            y = np.array(x, dtype=np.uint8)
            return np.sum(y) / y.size

        # Add text overlay by choosing random text, length, color and position
        while 1:
            font = ImageFont.truetype(serif, np.random.randint(16, 21))
            length = np.random.randint(10, 25)
            chars = ''.join(random.choice(ascii_letters) for _ in range(length))
            color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)

            # Update mask and check occupancy
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break

        return text_img


    def _corrupt(self, img):
        """Corrupts images (Gaussian, Poisson, or text overlay)."""

        if self.noise_type in ['gaussian', 'poisson']:
            return self._add_noise(img)
        elif self.noise_type == 'text':
            return self._add_text_overlay(img)
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])
        img =  Image.open(img_path).convert('RGB')

        # Random square crop
        if self.crop_size != 0:
            img = self._random_crop([img])[0]

        # Corrupt source image
        source = tvf.to_tensor(self._corrupt(img))

        # Corrupt target image, but not when clean targets are requested
        if self.clean_targets:
            target = tvf.to_tensor(img)
        else:
            target = tvf.to_tensor(self._corrupt(img))

        return source, target


class MonteCarloDataset(AbstractDataset):
    """Class for dealing with Monte Carlo rendered images."""

    def __init__(self, root_dir, redux, crop_size,
        hdr_buffers=False, hdr_targets=True, clean_targets=False):
        """Initializes Monte Carlo image dataset."""

        super(MonteCarloDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        # Rendered images directories
        self.root_dir = root_dir
        self.imgs = os.listdir(os.path.join(root_dir, 'render'))
        self.albedos = os.listdir(os.path.join(root_dir, 'albedo'))
        self.normals = os.listdir(os.path.join(root_dir, 'normal'))

        if redux:
            self.imgs = self.imgs[:redux]
            self.albedos = self.albedos[:redux]
            self.normals = self.normals[:redux]

        # Read reference image (converged target)
        ref_path = os.path.join(root_dir, 'reference.png')
        self.reference = Image.open(ref_path).convert('RGB')

        # High dynamic range images
        self.hdr_buffers = hdr_buffers
        self.hdr_targets = hdr_targets


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Use converged image, if requested
        if self.clean_targets:
            target = self.reference
        else:
            target_fname = self.imgs[index].replace('render', 'target')
            file_ext = '.exr' if self.hdr_targets else '.png'
            target_fname = os.path.splitext(target_fname)[0] + file_ext
            target_path = os.path.join(self.root_dir, 'target', target_fname)
            if self.hdr_targets:
                target = tvf.to_pil_image(load_hdr_as_tensor(target_path))
            else:
                target = Image.open(target_path).convert('RGB')

        # Get buffers
        render_path = os.path.join(self.root_dir, 'render', self.imgs[index])
        albedo_path = os.path.join(self.root_dir, 'albedo', self.albedos[index])
        normal_path =  os.path.join(self.root_dir, 'normal', self.normals[index])

        if self.hdr_buffers:
            render = tvf.to_pil_image(load_hdr_as_tensor(render_path))
            albedo = tvf.to_pil_image(load_hdr_as_tensor(albedo_path))
            normal = tvf.to_pil_image(load_hdr_as_tensor(normal_path))
        else:
            render = Image.open(render_path).convert('RGB')
            albedo = Image.open(albedo_path).convert('RGB')
            normal = Image.open(normal_path).convert('RGB')

        # Crop
        if self.crop_size != 0:
            buffers = [render, albedo, normal, target]
            buffers = [tvf.to_tensor(b) for b in self._random_crop(buffers)]
        else:
            buffers = [render, albedo, normal, target]
            buffers = [tvf.to_tensor(b) for b in buffers]

        # Stack buffers to create input volume
        source = torch.cat(buffers[:3], dim=0)
        target = buffers[3]

        return source, target


class IntrinsicNoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, redux, crop_size, image_extension='.png',
                 seed=None, target_averaging=False, random_order=False):
        """Initializes noisy image dataset."""
        self.image_extension = image_extension

        super(IntrinsicNoisyDataset, self).__init__(root_dir, redux, crop_size, False)

        self.image_groups = []

        self._find_files()

        self.interfaces_found = \
            all([ all([ 'interfaces' in entry.keys() for entry in group]) for group in self.image_groups ])

        self.target_averaging = target_averaging
        self.random_order = random_order

        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)

    def _find_files(self):
        # first find all images
        all_files = glob.glob(
            os.path.join(self.root_dir, '**/*' + self.image_extension), recursive=True)

        # extract relative filename and group name and number from the filename
        file_details = [self._filename_processing(filename) for filename in all_files]

        # sort the files by their relative filename
        file_details = natsorted(file_details, key=itemgetter(1))

        # then group them by their group name
        self.image_groups = [
            [{subelem[4]:subelem[0] for subelem in elem} for _, elem in groupby(group, key=itemgetter(3))]
             for key, group in groupby(file_details, key=itemgetter(2))]

    def _filename_processing(self, filename):
        relative_filename = filename.replace(self.root_dir, '')
        if relative_filename.startswith(os.path.sep):
            relative_filename = relative_filename[len(os.path.sep):]
        basename_parts = re.match(r'(.+?)_([0-9]+)(?:_([a-z]+))?' + re.escape(self.image_extension), os.path.split(filename)[1])
        basename = basename_parts[1]
        group_number = basename_parts[2]
        suffix = basename_parts[3]
        if suffix is None:
            suffix = 'image'

        return filename, relative_filename, basename, group_number, suffix

    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        group = self.image_groups[index]

        # there are just two images
        if self.random_order:
            target_indices = np.random.permutation(len(group))
        else:
            target_indices = range(len(group))

        source = Image.open(group[target_indices[0]]['image'])
        target = Image.open(group[target_indices[1]]['image'])
        if len(group) > 2 and self.target_averaging:
            target = tvf.to_tensor(target)
            for nextImageIndex in range(2, len(group)):
                next_target_image = Image.open(group[target_indices[nextImageIndex]]['image'])
                target = target + tvf.to_tensor(next_target_image)
            target = tvf.to_pil_image(target / float(len(group) - 1))
        if self.interfaces_found:
            target_interfaces = Image.open(group[target_indices[1]]['interfaces'])
        else:
            target_interfaces = None

        if self.crop_size != 0:
            if target_interfaces is not None:
                source, target, target_interfaces = self._random_crop([source, target, target_interfaces])
            else:
                source, target = self._random_crop([source, target])

        source = tvf.to_tensor(source)
        target = tvf.to_tensor(target)
        if target_interfaces is not None:
            target_interfaces = tvf.to_tensor(target_interfaces)

        if target_interfaces is not None:
            return source, {'image': target, 'interfaces': target_interfaces}
        else:
            return source, target

    def __len__(self):
        """Returns length of dataset."""

        return len(self.image_groups)
