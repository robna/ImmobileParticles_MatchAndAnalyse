import numpy as np
import pandas as pd
import cv2
from typing import List
from scipy import ndimage as ndi
from skimage import filters, io
from skimage.morphology import binary_closing, disk
from outputs import npImgArray_to_base64png, getSnipImage


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


def getContourRadii(contours: List[np.ndarray]) -> List[float]:
    """
    Calculates Contour Sphere Equivalent Radii in pixels, according to A = pi * r^2 -> r = sqrt(A / pi)
    """
    radii: List[float] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        radii.append(np.sqrt(area / np.pi))
    return radii


def measure_particles(grayImg: np.ndarray, contours: List[np.ndarray], um_per_px=1) -> pd.DataFrame:
    """Calculate Area, Perimeter and avg. Intensity in correct order."""
    areas, perimeters, intensities, snips, snipWs, snipHs = [], [], [], [], [], []
    for cnt in contours:
        areas.append(cv2.contourArea(cnt))
        perimeters.append(cv2.arcLength(cnt, closed=True))
        mask = np.zeros(grayImg.shape, np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        intensities.append(cv2.mean(grayImg, mask=mask)[0])  # we only have grayscale, so only take first index [0]

        snipImage = getSnipImage(cnt, grayImg, um_per_px)
        snipHs.append(snipImage.shape[0])
        snipWs.append(snipImage.shape[1])
        snip64formatted = npImgArray_to_base64png(snipImage)  # convert extracted snip image to base64 encoded png
        snips.append(snip64formatted)  # save extracted particle image as base64 encoded png
        # snip.append([snipslice])  # alternative: save extracted particle image as array

    dataframe = pd.DataFrame()
    dataframe["area"] = areas
    dataframe["perimeter"] = perimeters
    dataframe["intensity"] = intensities
    dataframe["snip"] = snips
    dataframe["snipW"] = snipWs
    dataframe["snipH"] = snipHs
    return dataframe
