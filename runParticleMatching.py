from skimage import io
import time
from typing import TYPE_CHECKING
from alignImages import *
from detection_hysteresis import getLowerThresholdForImg, RecognitionParameters, measure_particles
from outputs import generateOutputGraphs  # , getRatioOfProperty

if TYPE_CHECKING:
    import pandas as pd

t0 = time.time()

# TODO: change from px to µm as unit for calculations
px_res_orig = 0.35934295644272635  # µm / px in original resolution of microscope
pyr_lev = 2  # used image level of the pyramid image data from CZI images
px_res = px_res_orig * pyr_lev  # pixel resolution [µm / px] of the images that were effectively used

config = {"imgScaleFactor": 1,  # (0...1.0)
          "minParticleArea": 152,  # ~ ESD 10 µm     # in px**2  TODO: is "squarepixel" correct here? shouldn't it just be "pixel"
          "maxParticleArea": 15206,  # ~ ESD 100 µm  # in px**2
          "hystHighThresh": 0.75,  # relative to maximum intensity
          "particleDistTolerance": 3,  # in percent (0...100)
          "property": "area",  # the property to calculate the after/before ratio of
          "showPartImages": False  # whether or not to show the found and paired particles in before and after image
          }


def runPM(pathBeforeImg, pathAfterImg):
    beforeImg: np.ndarray = io.imread(pathBeforeImg)
    afterImg: np.ndarray = io.imread(pathAfterImg)

    beforeImg = cv2.resize(beforeImg, None, fx=config["imgScaleFactor"], fy=config["imgScaleFactor"])
    afterImg = cv2.resize(afterImg, None, fx=config["imgScaleFactor"], fy=config["imgScaleFactor"])

    beforeImg = cv2.medianBlur(beforeImg, ksize=9)
    afterImg = cv2.medianBlur(afterImg, ksize=9)

    params: RecognitionParameters = RecognitionParameters()
    params.highTreshold = config["hystHighThresh"]
    params.lowerThreshold = getLowerThresholdForImg(beforeImg)  # * 2
    params.minArea = config["minParticleArea"]
    params.maxArea = config["maxParticleArea"]
    params.doHysteresis = False  # Don't do hysteresis first, just to get all particles (better for getting transform)

    _, beforeContours = getLabelsAndContoursFromImage(beforeImg, params)
    beforeCenters: np.ndarray = getContourCenters(beforeContours)

    _, afterContours = getLabelsAndContoursFromImage(afterImg, params)
    afterCenters: np.ndarray = getContourCenters(afterContours)

    ymin, ymax = beforeCenters[:, 1].min(), beforeCenters[:, 1].max()
    maxDistError = (ymax - ymin) * config["particleDistTolerance"] / 100
    angle, shift = findAngleAndShift(beforeCenters, afterCenters, maxDistError)
    # print(f'angle: {angle}, shift: {shift}, numSrcCenters: {len(beforeContours)}, numDstCenters: {len(afterCenters)}')

    # No get only the relevant particles
    params.doHysteresis = True
    beforeLabels, beforeContours = getLabelsAndContoursFromImage(beforeImg, params)
    beforeCenters = getContourCenters(beforeContours)
    afterLabels, afterContours = getLabelsAndContoursFromImage(afterImg, params)
    afterCenters = getContourCenters(afterContours)

    transformedBefore: np.ndarray = offSetPoints(beforeCenters, angle, shift)
    # TODO: what is 'error' for?
    error, indexBefore2After = getIndicesAndErrosFromCenters(transformedBefore, afterCenters, maxDistError)

    statsBefore: 'pd.DataFrame' = measure_particles(beforeImg, beforeLabels)
    statsAfter: 'pd.DataFrame' = measure_particles(afterImg, afterLabels)
    # Ratios are now calculated in the result_plots notebook, so here I commented it out to save computation time
    # ratios: np.ndarray = getRatioOfProperty(config["property"], statsBefore, statsAfter, indexBefore2After)

    if config["showPartImages"]:
        fig1, fig2 = generateOutputGraphs(beforeCenters, afterCenters, beforeContours, afterContours, beforeImg,
                                          afterImg, 'before', 'after', indexBefore2After)
        fig1.show()
        fig2.show()

    return statsBefore, statsAfter, indexBefore2After  # , ratios
