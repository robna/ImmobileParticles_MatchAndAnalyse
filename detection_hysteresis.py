from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
from scipy import ndimage as ndi
from skimage import filters, img_as_ubyte, io
from skimage.measure import regionprops_table
from skimage.color import rgb2gray
from skimage.morphology import binary_closing, disk, remove_small_objects

import datetime


def identify_particles(im, disk_size=3, high=None):
    """
    Particles on input image are identified by hysteresis thresholding and labeled
    """
    if high is None:
        high = im.max() * 0.95
    else:
        high *= im.max()

    low = filters.threshold_li(im[im > 0])
    # The following is basically the manual way of ´hyst = filters.apply_hysteresis_threshold(im, low, high)´,
    # but includes some manipulations to the threshold masks (filling holes, removing small objects)
    mask_low = ndi.binary_fill_holes(im > low)

    selem = disk(disk_size)  # * scaling))
    binary_closing(mask_low, selem, out=mask_low)  # dilation followed by erosion to connect small speckles to particles
    mask_high = im >= high

    # Labeling connected components of mask_low (excluding objects touching image borders)
    labels_low, num_labels_low = ndi.label(mask_low)
    # remove_small_objects(labels_low, min_size=68, in_place=True)
    # seg.clear_border(labels_low, in_place=True)  # OBS! Can this be used for label ojects? It seems to break things for some images but not for others...

    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels_low + 1))
    connected_to_high = sums > 0
    hyst = connected_to_high[labels_low]
    hyst = ndi.binary_fill_holes(hyst)

    # Assigning labels to final hysteresis image
    labels_hyst, num_labels_hyst = ndi.label(hyst)
    return labels_hyst, num_labels_hyst, hyst, mask_low, mask_high, high, low


def measure_particles(im, labels_hyst):
    """
    Particles on input image are measured using the labeled regions and results are reported in a pandas dataframe
    """
    props = regionprops_table(labels_hyst, im, properties=(
        'label',
        'area',
        'perimeter',
        'major_axis_length',
        'minor_axis_length',
        'mean_intensity',
        'centroid',
        'bbox'))
    df = pd.DataFrame(props)
    return df


if __name__ == '__main__':
    # Pixel sizes of image
    orgPxSize = 0.359343 #this is the length (and width) of one pixel in µm in the original CZI images in full resolution

    # scaling = 0.1 #this is to downscale images for faster processing (set to 1.0 for no downscaling)
    df = pd.DataFrame()  # initialise empty dataframe for saving per-image particle measurments
    DF = pd.DataFrame()  # initialise empty dataframe collecting particle measurements from all images
    keys = pd.read_csv('wafer-polymer-keyfile.csv',index_col='wafer') # table containing polymer type information for each wafer

    # paths = pathlib.Path(r'C:\Users\xbrjos\Desktop\Bilderkennung Robin\TiffImages').rglob('*.tif')
    # paths = sorted([x for x in paths])

    paths = [r'TiffImages\w02a_pre.tif',
             r'TiffImages\w02a_water.tif']

    print("Startet at:   ", f"{datetime.datetime.now().time().isoformat('seconds'):<10}")

    for current_image_path in paths:
        cwn = current_image_path.stem.split('_')[0]  # get current wafer name
        cwp = keys.loc[cwn]['polymer']  # get current wafer polymer
        cwt = current_image_path.stem.split('_')[1]  # get current wafer treatment

        im = io.imread(current_image_path)
        labels_hyst, num_labels_hyst, hyst, mask_low, mask_high, high, low = identify_particles(im)

        df = measure_particles(im, labels_hyst)

        df.insert(0, 'file', current_image_path.name)
        df.insert(0, 'state', cwt)
        df.insert(0, 'polymer', cwp)
        df.insert(0, 'wafer', cwn)

        # Output:
        DF = DF.append(df)

        with open('quant_particle_data.csv', 'a') as output_csv:
            df.to_csv(output_csv, index=False, header=False)

        savePath = os.path.join(str(current_image_path.parent), 'jupyter_output', current_image_path.stem)
        io.imsave(savePath + "_jupyter_hyst.tiff", np.uint8(hyst))
        # io.imsave(savePath + "_jupyter_low.tiff", img_as_ubyte(mask_low))
        # io.imsave(savePath + "_jupyter_high.tiff", img_as_ubyte(mask_high))

        print(
            f"{datetime.datetime.now().time().isoformat('seconds'):<10}  --> ",
            f"{current_image_path.stem:<15}",
            "Number of particles = {num_labels_hyst:>6}",
            " low thresh = {round(low, 2):^15}",
            " high thresh = {round(high, 2):^15}")

    print("Finished at:   ", f"{datetime.datetime.now().time().isoformat('seconds'):<10}")