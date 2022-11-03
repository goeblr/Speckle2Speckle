import PIL.Image
import os.path
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import MultipleLocator
# from grid_strategy import strategies
import numpy as np
import cv2
import pandas as pd
import itertools


def load_image(path, suffix = None):
    if suffix is not None:
        path = path.format(suffix)
    return PIL.Image.open(path)


def collect_input_images(base_path, image_numbers, phantom_instances, type=None):
    # imageCompressed_speckle_instance_<image_number>_<phantom_instance>.png
    # or
    # imageCompressed_speckle_instance_<image_number>_<phantom_instance>_type.png
    image_filenames = [x for x in itertools.product(image_numbers, phantom_instances)]
    if type is not None:
        image_filenames = [(x, os.path.join(base_path,
                                            "imageCompressed_speckle_instance_{}_{}_{}.png".format(x[0], x[1], type)))
                           for x in image_filenames]
    else:
        image_filenames = [(x, os.path.join(base_path,
                                            "imageCompressed_speckle_instance_{}_{}.png".format(x[0], x[1])))
                           for x in image_filenames]
    return [(p[0], PIL.Image.open(p[1])) for p in image_filenames]


def average_images(image_list):
    assert(len(image_list) >= 1)
    mean_image = image_list[0]

    for image_index in range(1, len(image_list)):
        mean_image = mean_image + image_list[image_index]

    mean_image = mean_image / len(image_list)
    return mean_image


def blend_images(image1, image2, blending_options):
    if image1.dtype == np.uint8:
        image1 = np.array(image1).astype(float) / 255.0
        image2 = np.array(image2).astype(float) / 255.0

    image1_scaled = (image1 - blending_options['image1_range'][0]) / (
            blending_options['image1_range'][1] - blending_options['image1_range'][0])
    image2_scaled = (image2 - blending_options['image2_range'][0]) / (
                blending_options['image2_range'][1] - blending_options['image2_range'][0])
    image_blended = image1_scaled * blending_options['blending'] + image2_scaled * (1.0 - blending_options['blending'])

    image1 = np.minimum(np.maximum(image1 * 255.0, 0.0), 255.0).astype(np.uint8)
    image2 = np.minimum(np.maximum(image2 * 255.0, 0.0), 255.0).astype(np.uint8)
    image_blended = np.minimum(np.maximum(image_blended * 255.0, 0.0), 255.0).astype(np.uint8)
    return image1, image2, image_blended


def save_image(image, path):
    if image.dtype != np.uint8:
        image = np.minimum(np.maximum(image * 255.0, 0.0), 255.0).astype(np.uint8)

    if isinstance(path, tuple) or isinstance(path, list):
        fullpath = os.path.join(*path)
    else:
        fullpath = path
    f = open(fullpath, 'wb')
    image = PIL.Image.fromarray(image)
    image.save(fullpath, 'png')
    f.close()


def compute_mse(reference, to_asses):
    assert(reference[0][0] == to_asses[0][0])
    mse = np.mean((reference[1] - to_asses[1]) ** 2)
    return mse


def compute_mses(reference, to_asses):
    mses = [compute_mse(x[0], x[1]) for x in zip(reference, to_asses)]
    return mses


def compute_mad(reference, to_asses):
    assert(reference[0][0] == to_asses[0][0])
    mad = np.mean(np.abs(reference[1] - to_asses[1]))
    return mad


def compute_mads(reference, to_asses):
    mads = [compute_mad(x[0], x[1]) for x in zip(reference, to_asses)]
    return mads


def extract_rect(image, rect):
    # rect is given as [xmin ymin width height]
    x = rect[0]
    y = rect[1]
    x_end = x + rect[2]
    y_end = y + rect[3]
    image_r = image[y:y_end, x:x_end]
    return image_r.flatten()


def compute_std(image, rect):
    image_r = extract_rect(image, rect)
    std = np.std(image_r)
    return std


def mark_rect(image, rect, rect_options):
    # rect = [xmin ymin width height]
    if isinstance(image, np.ndarray) and image.dtype == np.double:
        image = image.astype(np.float32)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),
                          color=rect_options['color'], thickness=rect_options['linewidth'])
    return image


if __name__ == "__main__":
    metrics_to_compute = ['simulated_objects', 'experimental', 'invivo']

    plot_order = ['input', 'SRAD', 'MED', 'BILAT', 'NLM', 'OBNLM', 'speckle2speckle']
    plot_labels = {'input': 'Input', 'SRAD': 'SRAD', 'MED': 'Median', 'BILAT': 'Bilat.', 'NLM': 'NLM',
                   'OBNLM': 'OBNLM', 'speckle2speckle': 'Ours'}

    rect_options = {
        'color': (255, 255, 0),
        'linewidth': 2
    }
    blend_options = {
        'blending': 0.75, # 75% filtered, 25% original
        'image1_range': (0.1, 0.9),
        'image2_range': (0.0, 1)
    }

    if 'simulated_objects' in metrics_to_compute:
        image_numbers = range(1, 101)
        phantom_instances_input_average = range(2, 11)
        
        path_speckle2speckle = "../outputImages/simulated"
        suffix_speckle2speckle = "denoised"
        path_inputs = "../data/test-set_simulated"
        path_other_methods = "../outputImages/otherMethods_simulated"
        suffixes_other_methods = ["BILAT", "MED", "NLM", "OBNLM", "SRAD"]

        images = {
            'input': collect_input_images(path_inputs, image_numbers, [1]),
            'input_average': collect_input_images(path_inputs, image_numbers, phantom_instances_input_average),
            'speckle2speckle': collect_input_images(path_speckle2speckle, image_numbers, [1], suffix_speckle2speckle)}

        for other_method in suffixes_other_methods:
            images[other_method] = collect_input_images(path_other_methods, image_numbers, [1], other_method)

        for key in images.keys():
            images[key] = [(x[0], np.array(x[1]).astype(float) / 255.0) for x in images[key]]

        images['average'] = []
        for _, group in itertools.groupby(images['input_average'], lambda x: x[0][0]):
            all_images = [x for x in group]
            new_key = all_images[0][0]
            in_images = [x[1] for x in all_images]
            ave_image = average_images(in_images)

            images['average'].append((new_key, ave_image))

        del images['input_average']

        mses_table = {}
        mads_table = {}
        statistics = []
        for key in images.keys():
            if key != 'average':
                mses = compute_mses(images['average'], images[key])
                mads = compute_mads(images['average'], images[key])
                statistics.append({'method': key,
                                   'mean_mse': np.mean(mses), 'stdev_mse': np.std(mses),
                                   'mean_mad': np.mean(mads), 'stdev_mad': np.std(mads)})
                mses_table[key] = mses
                mads_table[key] = mads
            else:
                image_nums = [x[0][0] for x in images[key]]
                mses_table['image_num'] = image_nums
                mads_table['image_num'] = image_nums

        mses_frame = pd.DataFrame(mses_table)
        mses_frame.to_csv('../simulated_objects_mses.csv', index=False)
        mads_frame = pd.DataFrame(mads_table)
        mads_frame.to_csv('../simulated_objects_mads.csv', index=False)
        statistics_frame = pd.DataFrame(statistics)
        statistics_frame.to_csv('../simulated_objects_statistics.csv', index=False)
        statistics_frame.to_latex('../simulated_objects_statistics.tex', index=False, float_format="%.2e")

        plt.figure()
        plt.boxplot([mses_table[x] for x in mses_table.keys() if x != 'image_num'],
                    labels=[x for x in mses_table.keys() if x != 'image_num'])

        plt.figure()
        plt.boxplot([mads_table[x] for x in mads_table.keys() if x != 'image_num'],
                    labels=[x for x in mads_table.keys() if x != 'image_num'])

        print(len(images))

    if 'experimental' in metrics_to_compute:
        path_speckle2speckle = "../outputImages/experimental/phantom_1_denoised.png"
        path_input = "../data/phantom/phantom_1.png"
        path_other_methods = "../outputImages/otherMethods_experimental/phantom_1_{}.png"
        suffixes_other_methods = ["BILAT", "MED", "NLM", "OBNLM", "SRAD"]

        # [xmin ymin width height]
        evaluation_rects = {'bg': [349, 348, 141, 142], 'inclusion': [150, 381, 64, 64]}

        images = {
            'input': load_image(path_input),
            'speckle2speckle': load_image(path_speckle2speckle)}

        for other_method in suffixes_other_methods:
            images[other_method] = load_image(path_other_methods, other_method)

        for key in images.keys():
            images[key] = np.array(images[key]).astype(float) / 255.0

        stds = {'region': {x: x for x in evaluation_rects.keys()}}
        rect_values = {}
        for key in images.keys():
            stds[key] = {}
            rect_values[key] = {}
            for region in evaluation_rects.keys():
                stds[key][region] = compute_std(images[key], evaluation_rects[region])
                rect_values[key][region] = extract_rect(images[key], evaluation_rects[region])

        # create violin plots
        labels = [plot_labels[x] for x in plot_order]
        for region in evaluation_rects.keys():
            fig = plt.figure()
            ax = plt.gca()
            values_region = [rect_values[key][region] for key in plot_order]
            parts = plt.violinplot(values_region, widths=0.9)
            for pc in parts['bodies']:
                color_before = pc.get_facecolor()[0]
                alpha_before = color_before[3]
                color_edge = color_before[0:3]
                color_face = [(1 - alpha_before) + color_before[0] * alpha_before,
                              (1 - alpha_before) + color_before[1] * alpha_before,
                              (1 - alpha_before) + color_before[2] * alpha_before]
                pc.set_alpha(1)
                pc.set_facecolor(color_face)
                pc.set_edgecolor(color_edge)

            plt.ylabel("Intensity")
            ax.xaxis.set_tick_params(direction='out')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_xlim(0.25, len(labels) + 0.75)
            ax.set_ylim(0.0, 0.9)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))

            fig.set_size_inches(4.6, 2)
            fig.tight_layout()
            plt.savefig('../plots/intensity_violin_{}_{}.pdf'.format('experimental', region))
            #plt.show()

        # mark the rects
        marked_input = images['input'].copy()
        for region in evaluation_rects.keys():
            marked_input = mark_rect(marked_input, evaluation_rects[region], rect_options)
        save_image(marked_input, "../plots/rect_{}".format(os.path.basename(path_input)))

        stds_frame = pd.DataFrame(stds)
        stds_frame.to_csv('../experimental_statistics.csv', index=False)
        stds_frame.to_latex('../experimental_statistics.tex', index=False, float_format="%.2e")

    if 'invivo' in metrics_to_compute:
        path_speckle2speckle = "../outputImages/invivo/invivo_1_denoised.png"
        path_input = "../data/invivo/invivo_1.png"
        path_other_methods = "../outputImages/otherMethods_invivo/invivo_1_{}.png"
        suffixes_other_methods = ["BILAT", "MED", "NLM", "OBNLM", "SRAD"]

        # [xmin ymin width height]
        evaluation_rects = {'thyroid': [278, 171, 109, 129]}

        blend_methods = ['speckle2speckle', 'OBNLM']

        images = {
            'input': load_image(path_input),
            'speckle2speckle': load_image(path_speckle2speckle)}

        for other_method in suffixes_other_methods:
            images[other_method] = load_image(path_other_methods, other_method)

        for key in images.keys():
            images[key] = np.array(images[key]).astype(float) / 255.0

        stds = {'region': {x: x for x in evaluation_rects.keys()}}
        rect_values = {}
        for key in images.keys():
            stds[key] = {}
            rect_values[key] = {}
            for region in evaluation_rects.keys():
                stds[key][region] = compute_std(images[key], evaluation_rects[region])
                rect_values[key][region] = extract_rect(images[key], evaluation_rects[region])

        # create violin plots
        labels = [plot_labels[x] for x in plot_order]
        for region in evaluation_rects.keys():
            fig = plt.figure()
            ax = plt.gca()
            values_region = [rect_values[key][region] for key in plot_order]
            parts = plt.violinplot(values_region, widths=0.9)
            for pc in parts['bodies']:
                color_before = pc.get_facecolor()[0]
                alpha_before = color_before[3]
                color_edge = color_before[0:3]
                color_face = [(1 - alpha_before) + color_before[0] * alpha_before,
                              (1 - alpha_before) + color_before[1] * alpha_before,
                              (1 - alpha_before) + color_before[2] * alpha_before]
                pc.set_alpha(1)
                pc.set_facecolor(color_face)
                pc.set_edgecolor(color_edge)

            plt.ylabel("Intensity")

            ax.xaxis.set_tick_params(direction='out')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels)
            ax.set_xlim(0.25, len(labels) + 0.75)
            ax.set_ylim(0.0, 0.9)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))

            fig.set_size_inches(4.6, 2)
            fig.tight_layout()
            plt.savefig('../plots/intensity_violin_{}_{}.pdf'.format('invivo', region))
            #plt.show()

        # mark the rects
        marked_input = images['input'].copy()
        for region in evaluation_rects.keys():
            marked_input = mark_rect(marked_input, evaluation_rects[region], rect_options)
        save_image(marked_input, "../plots/rect_{}".format(os.path.basename(path_input)))

        # create the blended images
        for blend_method in blend_methods:
            _, _, image_blended = blend_images(images[blend_method], images['input'], blend_options)
            save_image(image_blended, "../outputImages/blend_{}".format(os.path.basename(path_other_methods.format(blend_method))))

        stds_frame = pd.DataFrame(stds)
        stds_frame.to_csv('../invivo_statistics.csv', index=False)
        stds_frame.to_latex('../invivo_statistics.tex', index=False, float_format="%.2e")

    # create colorbar pdf
    a = np.zeros([2, 2])
    a[0, 0] = -70
    fig = plt.figure()
    im = plt.imshow(a, cmap='gray')
    bar = plt.colorbar()
    bar.ax.set_title('[dB]', fontsize=11)
    im.remove()
    ax = fig.gca()
    ax.remove()
    fig.set_size_inches(4.6, 3)
    fig.tight_layout()
    plt.savefig('../plots/colorbar.pdf')