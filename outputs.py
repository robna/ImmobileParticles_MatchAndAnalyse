from typing import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import colorsys
import base64
if TYPE_CHECKING:
    import pandas as pd


def generateOutputGraphs(sourceCenters: np.ndarray, dstCenters: np.ndarray,
                         sourceContours: List[np.ndarray], dstContours: List[np.ndarray],
                         srcImg: np.ndarray, dstImg: np.ndarray,
                         srcName: str, dstName: str,
                         indicesBeforeAfter: Dict[int, int]) -> Tuple[plt.figure, plt.figure]:

    srcImg = srcImg.copy()
    srcImg = cv2.cvtColor(srcImg, cv2.COLOR_GRAY2RGB)
    srcimgMarkovers = np.zeros_like(srcImg)
    dstImg = dstImg.copy()
    dstImg = cv2.cvtColor(dstImg, cv2.COLOR_GRAY2RGB)
    dstImgMarkovers = np.zeros_like(dstImg)

    # now fill matched particles with color
    numParticlesMatched = len(indicesBeforeAfter)
    colorStep = 1 / numParticlesMatched
    hue = 0
    for origInd, targetInd in indicesBeforeAfter.items():
        color = colorsys.hsv_to_rgb(hue, 1, 1)
        color = color[0] * 255, color[1] * 255, color[2] * 255

        x, y = int(round(sourceCenters[origInd, 0])), int(round(sourceCenters[origInd, 1]))
        cnt = sourceContours[origInd]
        cv2.drawContours(srcimgMarkovers, [cnt], -1, color, -1)

        x, y = int(round(dstCenters[targetInd, 0])), int(round(dstCenters[targetInd, 1]))
        cnt = dstContours[targetInd]
        cv2.drawContours(dstImgMarkovers, [cnt], -1, color, -1)

        hue += colorStep

    srcImg = np.uint8(np.round(0.5 * srcImg + 0.5 * srcimgMarkovers))
    dstImg = np.uint8(np.round(0.5 * dstImg + 0.5 * dstImgMarkovers))

    # draw all contours with red outline
    cv2.drawContours(srcImg, sourceContours, -1, (255, 0, 0), 2)
    cv2.drawContours(dstImg, dstContours, -1, (255, 0, 0), 2)

    # now add numbers
    # fontSize = int(round(srcImg.shape[0] / 300))
    # for origInd, targetInd in indicesBeforeAfter.items():
    #     x, y = int(round(sourceCenters[origInd, 0])), int(round(sourceCenters[origInd, 1]))
    #     cv2.putText(srcImg, str(origInd), (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), thickness=fontSize)
    #
    #     x, y = int(round(dstCenters[targetInd, 0])), int(round(dstCenters[targetInd, 1]))
    #     cv2.putText(dstImg, str(origInd), (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), thickness=fontSize)

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot()
    # ax1.set_title(f'{len(indicesBeforeAfter)} of {sourceCenters.shape[0]} particles on {srcName} treatment image')
    # ax1.imshow(srcImg, cmap='gray')
    # fig1.tight_layout()
    #
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot()
    # ax2.set_title(f'{len(indicesBeforeAfter)} of {dstCenters.shape[0]} particles on {dstName} treatment image')
    # ax2.imshow(dstImg, cmap='gray')
    # fig2.tight_layout()
    # return fig1, fig2
    return srcImg, dstImg


def getRatioOfProperty(prop: str, dfBefore: 'pd.DataFrame', dfAfter: 'pd.DataFrame', indicesBeforeAfter: Dict[int, int]) -> np.ndarray:
    ratios: np.ndarray = np.zeros((len(indicesBeforeAfter), ))  # property before / property after for each index

    assert prop in dfBefore.columns and prop in dfAfter.columns, f'Requested property {prop} not found in dataframe'
    propBefore: pd.Series = dfBefore.get(prop)
    propAfter: pd.Series = dfAfter.get(prop)

    for i, (origInd, targetInd) in enumerate(indicesBeforeAfter.items()):
        ratios[i] = propBefore[origInd] / propAfter[targetInd]

    return ratios

def npImgArray_to_base64png(im):
    _, buffer = cv2.imencode('.png', np.ascontiguousarray(im))
    im64 = base64.b64encode(buffer).decode()
    im64formatted = 'data:image/png;base64,{}'.format(im64)

    return im64formatted