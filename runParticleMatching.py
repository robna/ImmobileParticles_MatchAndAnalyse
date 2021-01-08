from skimage import io
from skimage import exposure
import time
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
from alignImages import *
from detection_hysteresis import getLowerThresholdForImg, RecognitionParameters, measure_particles
from outputs import generateOutputGraphs, npImgArray_to_base64png,getRatioOfProperty

if TYPE_CHECKING:
    import pandas as pd

t0 = time.time()

# TODO: change from px to µm as unit for calculations
px_res_orig = 0.35934295644272635  # µm / px in original resolution of microscope
pyr_lev = 2  # used image level of the pyramid image data from CZI images
px_res = px_res_orig * pyr_lev  # pixel resolution [µm / px] of the images that were effectively used
minESD = 10
minArea_um = (minESD / 2) ** 2 * np.pi
minArea_px = round(minArea_um / px_res ** 2)
maxESD = 200
maxArea_um = (maxESD / 2) ** 2 * np.pi
maxArea_px = round(maxArea_um / px_res ** 2)

config = {"imgScaleFactor": 1.0,  # (0...1.0)
          "minParticleArea": minArea_px,
          # ~ ESD 10 µm     # in px**2  TODO: is "squarepixel" correct here? shouldn't it just be "pixel"
          "maxParticleArea": maxArea_px,  # ~ ESD 200 µm  # in px**2
          "hystHighThresh": 0.75,  # relative to maximum intensity
          "particleDistTolerance": 3,  # in percent (0...100)
          "property": "area",  # the property to calculate the after/before ratio of
          "showPartImages": True,  # whether or not to show the found and paired particles in before and after image
          "multiprocessing": False
          }


def runPM(pathBeforeImg, pathAfterImg):
    beforeImg: np.ndarray = io.imread(pathBeforeImg)
    afterImg: np.ndarray = io.imread(pathAfterImg)

    # beforeImg = exposure.match_histograms(beforeImg, afterImg).astype(np.uint8)

    beforeImg_nonBlur = cv2.resize(beforeImg, None, fx=config["imgScaleFactor"], fy=config["imgScaleFactor"])
    afterImg_nonBlur = cv2.resize(afterImg, None, fx=config["imgScaleFactor"], fy=config["imgScaleFactor"])

    beforeImg = cv2.medianBlur(beforeImg, ksize=9)
    afterImg = cv2.medianBlur(afterImg, ksize=9)

    params: RecognitionParameters = RecognitionParameters()
    params.highTreshold = config["hystHighThresh"]
    params.lowerThreshold = getLowerThresholdForImg(beforeImg)  # * 2
    params.minArea = config["minParticleArea"]
    params.maxArea = config["maxParticleArea"]
    params.doHysteresis = False  # Don't do hysteresis first, just to get all particles (better for getting transform)

    _, beforeContours, beforeLowerTH = getLabelsAndContoursFromImage(beforeImg, params)
    beforeCenters: np.ndarray = getContourCenters(beforeContours)

    _, afterContours, afterLowerTH = getLabelsAndContoursFromImage(afterImg, params)
    afterCenters: np.ndarray = getContourCenters(afterContours)

    beforeMaxBG = np.argmax(cv2.calcHist([beforeImg_nonBlur], [0], None, [256], [0, 256])[1:beforeLowerTH]) + 1  # find most abundant background grey value
    afterMaxBG = np.argmax(cv2.calcHist([afterImg_nonBlur], [0], None, [256], [0, 256])[1:afterLowerTH]) + 1
    beforeMaxFG = np.argmax(cv2.calcHist([beforeImg_nonBlur], [0], None, [256], [0, 256])[beforeLowerTH:-1]) + 128  # find most abundant foreground grey value
    afterMaxFG = np.argmax(cv2.calcHist([afterImg_nonBlur], [0], None, [256], [0, 256])[afterLowerTH:-1]) + 128

    beforeMax = (beforeMaxBG, beforeMaxFG)
    afterMax = (afterMaxBG, afterMaxFG)

    ymin, ymax = beforeCenters[:, 1].min(), beforeCenters[:, 1].max()
    maxDistError = (ymax - ymin) * config["particleDistTolerance"] / 100
    angle, shift = findAngleAndShift(beforeCenters, afterCenters, maxDistError)
    # print(f'angle: {angle}, shift: {shift}, numSrcCenters: {len(beforeContours)}, numDstCenters: {len(afterCenters)}')

    # No get only the relevant particles
    params.doHysteresis = True
    beforeLabels, beforeContours, *_ = getLabelsAndContoursFromImage(beforeImg, params)
    beforeCenters = getContourCenters(beforeContours)
    afterLabels, afterContours, *_ = getLabelsAndContoursFromImage(afterImg, params)
    afterCenters = getContourCenters(afterContours)

    transformedBefore: np.ndarray = offSetPoints(beforeCenters, angle, shift)
    _, indexBefore2After = getIndicesAndErrosFromCenters(transformedBefore, afterCenters, maxDistError)

    statsBefore: 'pd.DataFrame' = measure_particles(beforeImg_nonBlur, beforeContours, um_per_px=px_res)
    statsAfter: 'pd.DataFrame' = measure_particles(afterImg_nonBlur, afterContours, um_per_px=px_res)

    if config["showPartImages"] and beforeImg.size < 100_000_000 and afterImg.size < 100_000_000:  # check that image size does not exceed 100 MB
        # fig1, fig2 = generateOutputGraphs(beforeCenters, afterCenters, beforeContours, afterContours, beforeImg,
        #                                   afterImg, 'before', 'after', indexBefore2After)
        # fig1.show()
        # fig2.show()
        # plt.show(block=True)
        srcImg, dstImg = generateOutputGraphs(beforeCenters, afterCenters, beforeContours, afterContours, beforeImg,
                                          afterImg, 'before', 'after', indexBefore2After)

        def output_image_resizing(img):
            width = 800  # output image width should be 800 px
            scaling = width / img.shape[1]
            height = round(img.shape[0] * scaling)  # scale height proportionally
            dim = (width, height)
            return dim

        srcImg = cv2.resize(srcImg, output_image_resizing(srcImg), interpolation = cv2.INTER_AREA)
        dstImg = cv2.resize(dstImg, output_image_resizing(dstImg), interpolation = cv2.INTER_AREA)

        srcImg64formatted = npImgArray_to_base64png(srcImg)  # convert image to base64 encoded png
        dstImg64formatted = npImgArray_to_base64png(dstImg)  # convert image to base64 encoded png

        imgOverlays = {'pre_imgOverlay': srcImg64formatted, 'post_imgOverlay': dstImg64formatted}

    return statsBefore, statsAfter, indexBefore2After, beforeMax, afterMax, imgOverlays if 'imgOverlays' in locals() else None  # , ratios  # ratios not needed anymore (are calculated now in results notebook)

    # else:
    #     return statsBefore, statsAfter, indexBefore2After, beforeMax, afterMax  # , ratios
