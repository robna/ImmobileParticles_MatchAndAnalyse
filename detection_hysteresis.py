import numpy as np
import pandas as pd
import cv2
from typing import List
from scipy import ndimage as ndi
from skimage import filters, io
from skimage.morphology import binary_closing, disk
from outputs import npImgArray_to_base64png



class RecognitionParameters(object):
    def __init__(self):
        super(RecognitionParameters, self).__init__()
        self.highTreshold: float = 0.95  # from 0.0 to 1.0
        self.lowerThreshold: float = np.nan
        self.diskSize: int = 3
        self.minArea: float = 500
        self.maxArea: float = 1e6
        self.doHysteresis: bool = True


def getLowerThresholdForImg(grayscaleImg: np.ndarray) -> float:
    """
    calculates an optimized threshold for converting the grayscale image into a binary image.
    """
    return filters.threshold_otsu(grayscaleImg[grayscaleImg > 0])


def identify_particles(img: np.ndarray, parameters: 'RecognitionParameters' = RecognitionParameters()):
    """
    Particles on input image are identified by hysteresis thresholding and labeled
    The used params are also returned, in case they
    """
    high = img.max() * parameters.highTreshold
    if np.isnan(parameters.lowerThreshold):
        parameters.lowerThreshold = getLowerThresholdForImg(img)
    low = parameters.lowerThreshold
    # The following is basically the manual way of ´hyst = filters.apply_hysteresis_threshold(im, low, high)´,
    # but includes some manipulations to the threshold masks (filling holes, removing small objects)
    mask_low = ndi.binary_fill_holes(img > low)

    selem = disk(parameters.diskSize)  # * scaling))
    binary_closing(mask_low, selem, out=mask_low)  # dilation followed by erosion to connect small speckles to particles
    mask_high = img >= high

    # Labeling connected components of mask_low (excluding objects touching image borders)
    labels_low, num_labels_low = ndi.label(mask_low)

    # Check which connected components contain pixels from mask_high
    if parameters.doHysteresis:
        connected_to_high = np.zeros(num_labels_low + 1).astype(np.bool)
        for i in range(num_labels_low):
            if i > 0:
                ind = labels_low == i
                masked = img[ind].copy()
                numPxLabel = cv2.countNonZero(masked)
                masked[masked < high] = 0
                numPxHigh = cv2.countNonZero(masked)

                if numPxHigh / numPxLabel > 0.1:
                    connected_to_high[i] = True

        hyst = connected_to_high[labels_low]
    else:
        hyst = labels_low

    hyst = ndi.binary_fill_holes(hyst)
    # Assigning labels to final hysteresis image
    labels_hyst, num_labels_hyst = ndi.label(hyst)
    return labels_hyst, num_labels_hyst, hyst, mask_low, mask_high, high, low


def measure_particles(grayImg: np.ndarray, contours: List[np.ndarray], um_per_px=1) -> pd.DataFrame:
    """Calculate Area, Perimeter and avg. Intensity in correct order."""
    areas, perimeters, intensities, snips, snipWs, snipHs = [], [], [], [], [], []
    for cnt in contours:
        areas.append(cv2.contourArea(cnt))
        perimeters.append(cv2.arcLength(cnt, closed=True))
        mask = np.zeros(grayImg.shape, np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        intensities.append(cv2.mean(grayImg, mask=mask)[0])  # we only have grayscale, so only take first index [0]
        x, y, w, h = cv2.boundingRect(cnt)  # we get x, y, width and height of the bounding box for the contour
        snipWs.append(w)
        snipHs.append(h)
        snipslice = grayImg[
                    y - 0:y + h + 0,
                    x - 0:x + w + 0].astype('uint8')  # extract the part of the image that is within bb + X px each way
        SBl = 10  # length of scale bar in µm
        SBw = round(SBl / um_per_px)  # width of scale bar in px
        SBh = 5  # height of scale bar in px
        SBx = round(snipslice.shape[1]* 0.1)  # find the x position to start the scale bar
        SBy = round(snipslice.shape[0] * 0.9)  # find the y position to start the scale bar
        if SBx + SBw <= snipslice.shape[1]:  # test if scale bar would fit snip image width
            if snipslice[SBy - SBh:SBy, SBx:SBx + SBw].mean() <= 127:
                snipslice[SBy - SBh:SBy, SBx:SBx + SBw] = 255  # make white scale bar where background is dark
            else:  # make scale bar go across whole snip image if it does not fit
                snipslice[SBy - SBh:SBy, SBx:SBx + SBw] = 0  # make white scale bar where background is bright
        else:
            if snipslice[SBy - SBh:SBy, SBx:].mean() <= 127:
                snipslice[SBy - SBh:SBy, SBx:] = 255  # make white scale bar where background is dark
            else:
                snipslice[SBy - SBh:SBy, SBx:] = 0  # make white scale bar where background is bright
        snip64formatted = npImgArray_to_base64png(snipslice)  # convert extracted snip image to base64 encoded png
        snips.append(snip64formatted)  # save extracted particle image as base64 encoded png
        # snip.append([snipslice])  # alternative: save extracted particle image as array


    dataframe = pd.DataFrame()
    dataframe["area"] = areas
    dataframe["perimeter"] = perimeters
    dataframe["intensity"] = intensities
    dataframe["snip"] = snips
    dataframe["snipW"] = [W + 0 for W in snipWs]  # width plus 2 x padding, which was set when taking snipslice
    dataframe["snipH"] = [H + 0 for H in snipHs]  # height plus 2 x padding, which was set when taking snipslice
    return dataframe
