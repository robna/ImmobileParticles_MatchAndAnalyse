from skimage import io
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
from alignImages import *
from detection_hysteresis import getLowerThresholdForImg, RecognitionParameters, measure_particles
from outputs import generateOutputGraphs, getRatioOfProperty
if TYPE_CHECKING:
    import pandas as pd

t0 = time.time()

config = {# "pathBeforeImg": r'/run/media/nibor/data_ext/quantDigest_imageData/tif/w73_pre.tif',
          # "pathAfterImg": r'/run/media/nibor/data_ext/quantDigest_imageData/tif/w73_pentane.tif',
          "imgScaleFactor": 1,  # (0...1.0)
          "minParticleArea": 600,  # in px**2
          "maxParticleArea": np.inf,  # 15000,  # in px**2
          "hystHighThresh": 0.75,  # relative to maximum intensity
          "particleDistTolerance": 3,  # in percent (0...100)
          "property": "area",  # the property to calculate the after/before ratio of
          "showPartImages": False  # whether or not to show the found and paired particles in before and after image
          }

#afterName = os.path.basename(config["pathAfterImg"]).split('tif')[0]
#beforeName = os.path.basename(config["pathBeforeImg"]).split('tif')[0]

#beforeImg: np.ndarray = io.imread(config["pathBeforeImg"])
#afterImg: np.ndarray = io.imread(config["pathAfterImg"])

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
    # print(f'loading images and getting contours took {time.time()-t0} seconds')

    t0 = time.time()
    ymin, ymax = beforeCenters[:, 1].min(), beforeCenters[:, 1].max()
    maxDistError = (ymax - ymin) * config["particleDistTolerance"] / 100
    angle, shift = findAngleAndShift(beforeCenters, afterCenters, maxDistError)
    # print(f'angle: {angle}, shift: {shift}, numSrcCenters: {len(beforeContours)}, numDstCenters: {len(afterCenters)}')
    # print(f'getting transform and error took {time.time()-t0} seconds')

    t0 = time.time()
    # No get only the relevant particles
    params.doHysteresis = True
    beforeLabels, beforeContours = getLabelsAndContoursFromImage(beforeImg, params)
    beforeCenters = getContourCenters(beforeContours)
    afterLabels, afterContours = getLabelsAndContoursFromImage(afterImg, params)
    afterCenters = getContourCenters(afterContours)

    transformed: np.ndarray = offSetPoints(beforeCenters, angle, shift)
    error, indexMap = getIndicesAndErrosFromCenters(transformed, afterCenters, maxDistError)

    statsBefore: 'pd.DataFrame' = measure_particles(beforeImg, beforeLabels)
    statsAfter: 'pd.DataFrame' = measure_particles(afterImg, afterLabels)

    return statsBefore, statsAfter, indexMap

if __name__ == '__main__':
    print(f"--------------RESULTS OF COMPARING {afterName.upper()} to {beforeName.upper()}--------------")
    print(f'Num Particles before: {len(beforeContours)}, num Particles after: {len(afterContours)}')
    print(f'Thereof {len(indexMap)} could be successfully paired to calculate statistics.')

    if config["showPartImages"]:
        fig1, fig2 = generateOutputGraphs(beforeCenters, afterCenters, beforeContours, afterContours, beforeImg, afterImg,
                                          beforeName, afterName, indexMap)
        fig1.show()
        fig2.show()

    ratios: np.ndarray = getRatioOfProperty(config["property"], statsBefore, statsAfter, indexMap)
    resultFig: plt.Figure = plt.figure()
    ax: plt.Axes = resultFig.add_subplot()
    ax.boxplot(ratios, showfliers=False)
    ax.set_title(f"BoxPlot of {beforeName} / {afterName} \nratio of {config['property']} of {len(indexMap)} particles")
    resultFig.show()
    print(f"Mean ratio of {config['property']} = {np.mean(ratios)}")

