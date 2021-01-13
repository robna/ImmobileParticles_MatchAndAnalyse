from skimage import io
from skimage import exposure
import time
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
from alignImages import *
from detection_hysteresis import getLowerThresholdForImg, RecognitionParameters, measure_particles, getContourRadii
from outputs import generateOutputGraphs, npImgArray_to_base64png

if TYPE_CHECKING:
    import pandas as pd

t0 = time.time()


def getPxAreaOfEquivalentSphere(diameter: float, pxRes: float, scaleFactor: float) -> int:
    """
    :param diameter: Desired particle sphere equivalent diameter in µm
    :param pxRes: Pixel resolution in µm/px
    :param scaleFactor: Scale factor of the input image
    :return pxArea: Pixel area of equivalent sphere
    """
    scaledPxRes: float = pxRes / scaleFactor
    area_um: float = (diameter / 2) ** 2 * np.pi
    area_px: int = int(round(area_um / scaledPxRes**2))
    return area_px


def runParticleMatching(pathBeforeImg, pathAfterImg):
    from compareDigestImages import Config, px_res
    beforeImg: np.ndarray = io.imread(pathBeforeImg)
    afterImg: np.ndarray = io.imread(pathAfterImg)

    scaleFac: float = Config.imgScaleFactor
    beforeImg = cv2.resize(beforeImg, None, fx=scaleFac, fy=scaleFac)
    afterImg = cv2.resize(afterImg, None, fx=scaleFac, fy=scaleFac)
    beforeImg_nonBlur: np.darray = beforeImg.copy()
    afterImg_nonBlur: np.ndarray = afterImg.copy()

    beforeImg = cv2.medianBlur(beforeImg, ksize=9)
    afterImg = cv2.medianBlur(afterImg, ksize=9)

    params: RecognitionParameters = RecognitionParameters()
    params.highTreshold = Config.hystHighThresh
    params.lowerThreshold = getLowerThresholdForImg(beforeImg)  # * 2
    params.minArea = Config.minParticleArea
    params.maxArea = Config.maxParticleArea
    params.doHysteresis = False  # Don't do hysteresis first, just to get all particles (better for getting transform)

    _, beforeContours, beforeLowerTH = getLabelsAndContoursFromImage(beforeImg, params)
    beforeCenters: np.ndarray = getContourCenters(beforeContours)

    _, afterContours, afterLowerTH = getLabelsAndContoursFromImage(afterImg, params)
    afterCenters: np.ndarray = getContourCenters(afterContours)

    beforeMax, afterMax = getBeforeAfterMax(beforeImg_nonBlur, afterImg_nonBlur, beforeLowerTH, afterLowerTH)

    beforeRadii: List[float] = getContourRadii(beforeContours)
    angle, shift = findAngleAndShift(beforeCenters, afterCenters, beforeRadii)

    # Now get only the relevant particles
    params.doHysteresis = True
    beforeLabels, beforeContours, *_ = getLabelsAndContoursFromImage(beforeImg, params)
    beforeCenters = getContourCenters(beforeContours)
    afterLabels, afterContours, *_ = getLabelsAndContoursFromImage(afterImg, params)
    afterCenters = getContourCenters(afterContours)

    transformedBefore: np.ndarray = offSetPoints(beforeCenters, angle, shift)
    _, indexBefore2After = getIndicesAndErrosFromCenters(transformedBefore, afterCenters, beforeRadii)

    statsBefore: 'pd.DataFrame' = measure_particles(beforeImg_nonBlur, beforeContours, um_per_px=px_res)
    statsAfter: 'pd.DataFrame' = measure_particles(afterImg_nonBlur, afterContours, um_per_px=px_res)

    if Config.showPartImages and beforeImg.size < 100_000_000 and afterImg.size < 100_000_000:  # check that image size does not exceed 100 MB
        srcImg, dstImg = generateOutputGraphs(beforeCenters, afterCenters, beforeContours, afterContours, beforeImg,
                                              afterImg, 'before', 'after', indexBefore2After)

        def output_image_resizing(img):
            width = 800  # output image width should be 800 px
            scaling = width / img.shape[1]
            height = round(img.shape[0] * scaling)  # scale height proportionally
            dim = (width, height)
            return dim

        srcImg = cv2.resize(srcImg, output_image_resizing(srcImg), interpolation=cv2.INTER_AREA)
        dstImg = cv2.resize(dstImg, output_image_resizing(dstImg), interpolation=cv2.INTER_AREA)

        srcImg64formatted = npImgArray_to_base64png(srcImg)  # convert image to base64 encoded png
        dstImg64formatted = npImgArray_to_base64png(dstImg)  # convert image to base64 encoded png

        imgOverlays = {'pre_imgOverlay': srcImg64formatted, 'post_imgOverlay': dstImg64formatted}

    return statsBefore, statsAfter, indexBefore2After, beforeMax, afterMax, imgOverlays if 'imgOverlays' in locals() else None  # , ratios  # ratios not needed anymore (are calculated now in results notebook)


def getBeforeAfterMax(imgBefore: np.ndarray, imgAfter: np.ndarray, threshBefore: int, threshAfter: int) -> Tuple:
    """
    Get the most abundant background and foreground gray values in the images. It's recommended to use non-blurred images.
    :param imgBefore:
    :param imgAfter:
    :param threshBefore:
    :param threshAfter:
    :return Tuple: beforeMax: tuple[maxValBackground, maxValForeground]
    """
    beforeMaxBG = np.argmax(cv2.calcHist([imgBefore], [0], None, [256], [0, 256])[1:threshBefore]) + 1
    afterMaxBG = np.argmax(cv2.calcHist([imgAfter], [0], None, [256], [0, 256])[1:threshAfter]) + 1

    beforeMaxFG = np.argmax(cv2.calcHist([imgBefore], [0], None, [256], [0, 256])[threshBefore:-1]) + threshBefore
    afterMaxFG = np.argmax(cv2.calcHist([imgAfter], [0], None, [256], [0, 256])[threshAfter:-1]) + threshAfter

    beforeMax = [beforeMaxBG, beforeMaxFG]
    afterMax = [afterMaxBG, afterMaxFG]
    return beforeMax, afterMax
