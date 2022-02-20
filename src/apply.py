#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os.path
import glob
import pathlib
from noise2noise import Noise2Noise

from argparse import ArgumentParser
import SimpleITK as sitk
import cv2
import math
import PIL.Image


def parse_args():
    """Command-line argument parser for application."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Speckle2Speckle')

    # Data parameters
    parser.add_argument('-i', '--input-data', help='mhd sequence to read / image folder')
    parser.add_argument('-o', '--output-data', help='filename to write result to')
    parser.add_argument('--load-ckpt', help='load model checkpoint')
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--blending-factor', help="0 = denoised only, 1 = input only", default=0.2, type=float)
    parser.add_argument('--contrast-window-center',
                        help="Center of the contrast window to use for the denoised images (video only)",
                        default=0.5, type=float)
    parser.add_argument('--contrast-window-width',
                        help="Width of the contrast window to use for the denoised images (video only)",
                        default=0.9, type=float)

    return parser.parse_args()


def write_video_opencv(images_to_write, filename, video_padding=16):
    # assumes images_to_write to be in [0, 1]
    # images_to_write = (images_to_write * 255).astype(np.uint8)

    # pad the video to a multiple of video_padding pixels
    image_width = images_to_write.shape[2]
    image_height = images_to_write.shape[1]
    padding_total = [
        math.ceil(image_width / video_padding) * video_padding - image_width,
        math.ceil(image_height / video_padding) * video_padding - image_height]
    padding = [ (0, 0),
        (math.ceil(padding_total[1] / 2), math.floor(padding_total[1] / 2)),
         (math.ceil(padding_total[0] / 2), math.floor(padding_total[0] / 2))]
    images_to_write = np.pad(images_to_write, padding)

    # create video writer
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5,
                          (images_to_write.shape[2], images_to_write.shape[1]))

    for image_index in range(images_to_write.shape[0]):

        image = np.tile(np.expand_dims(images_to_write[image_index, :, :], -1), (1, 1, 3))
        #image = images_to_write[image_index, :, :]
        image = (image * 255).astype(np.uint8)

        # Display the resulting frame
        cv2.imshow('frame', image)
        cv2.waitKey(1)

        out.write(image)

    out.release()


def write_video_scaled(images_input, images_output, param, filename):
    # assumed images_input and images_denoised to be generally scaled to [0, 1] (with some outliers)
    # clamp input images (just in case)
    images_input = np.maximum(np.minimum(images_input, 1.0), 0.0)
    # scale denoised to selected contrast window
    images_blended = np.maximum(np.minimum(
        (images_output - param.contrast_window_center) / param.contrast_window_width + 0.5, 1.0), 0.0)
    # blend input and denoised
    images_blended = (1 - param.blending_factor) * images_blended + param.blending_factor * images_input
    # clamp pure output images
    images_output = np.maximum(np.minimum(images_output, 1.0), 0.0)

    # put all three side-by-side
    images_to_write = np.concatenate((images_input, images_output, images_blended), axis=2)

    # write the video
    write_video_opencv(images_to_write, filename)

def load_image(path):
    fullpath = os.path.join(*path)
    pil_image = PIL.Image.open(fullpath)
    return np.array(pil_image)


def write_image(image, path):
    if image.dtype != np.uint8:
        image = np.minimum(np.maximum(image * 255.0, 0.0), 255.0).astype(np.uint8)

    fullpath = os.path.join(*path)
    f = open(fullpath, 'wb')
    image = PIL.Image.fromarray(image)
    image.save(fullpath, 'png')
    f.close()


if __name__ == '__main__':
    """Applies Noise2Noise."""

    # Parse test parameters
    params = parse_args()

    # Initialize model and test
    params.noise_type = 'intrinsic'
    n2n = Noise2Noise(params, trainable=False)
    #params.redux = False
    #params.clean_targets = True
    #test_loader = load_dataset(params.data, 0, params, shuffled=False, single=True,
    #                           target_averaging=params.average_validation_targets,
    #                           random_order=params.random_group_order)
    n2n.load_model(params.load_ckpt)

    # check whether the input is a file or a directory
    if os.path.isdir(params.input_data):
        # find all input files and create the output filename for it
        input_files = glob.glob(params.input_data + "/**/*.mhd", recursive=True)
        if len(input_files) == 0:
            input_files = glob.glob(params.input_data + "/**/*.png", recursive=True)
            input_files = [input_file for input_file in input_files if "_interfaces.png" not in input_file and "_labels.png" not in input_file and "_1.png" in input_file]
        output_files = [input_file.replace(params.input_data, params.output_data) for input_file in input_files]
    else:
        input_files = [params.input_data]
        output_files = [params.output_data]

    for file_index in range(len(input_files)):
        # load the mhd
        print("{} \ {}".format(file_index + 1, len(input_files)))
        if input_files[file_index][-3:] == "mhd":
            reader = sitk.ImageFileReader()
            reader.SetImageIO("MetaImageIO")
            reader.SetFileName(input_files[file_index])
            input_file = reader.Execute()
            images = sitk.GetArrayFromImage(input_file)
            image_read_sitk = True
        elif input_files[file_index][-3:] == "png":
            images = load_image([input_files[file_index]])
            images = images.reshape((1,) + images.shape).astype(np.float32)
            image_read_sitk = False
        # clamp and normalize
        images = np.maximum(np.minimum(images, 255.0), 0.0) / 255.0

        images_denoised = n2n.apply(images, transpose=False)

        if image_read_sitk:
            result_img = sitk.GetImageFromArray(images_denoised)
            result_img.SetSpacing(reader.GetSpacing())
            result_img.SetOrigin(reader.GetOrigin())
            result_img.SetDirection(reader.GetDirection())

            writer = sitk.ImageFileWriter()
            p = pathlib.Path(os.path.dirname(output_files[file_index]))
            p.mkdir(parents=True, exist_ok=True)
            writer.SetFileName(output_files[file_index])
            writer.Execute(result_img)

        if images.shape[0] > 1:
            write_video_scaled(images, images_denoised, params, output_files[file_index][:-4] + ".avi")
        write_image(images[0],          [output_files[file_index][:-4] + "_original.png"])
        write_image(images_denoised[0], [output_files[file_index][:-4] + "_denoised.png"])

