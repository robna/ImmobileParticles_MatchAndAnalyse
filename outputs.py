from typing import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import colorsys
import base64
if TYPE_CHECKING:
    import pandas as pd


def prepareDefaultWaferDataFrame(keysDF: 'pd.DataFrame') -> 'pd.DataFrame':
    keysDF.assign(
        pre_count=np.nan,
        post_count=np.nan,
        matched_count=np.nan,
        pre_histPeaks=None,
        post_histPeaks=None,
        pre_image='',
        post_image='',
        process_time=np.nan)
    return keysDF


def generateOutputGraphs(sourceCenters: np.ndarray, dstCenters: np.ndarray,
                         sourceContours: List[np.ndarray], dstContours: List[np.ndarray],
                         srcImg: np.ndarray, dstImg: np.ndarray,
                         srcName: str, dstName: str,
                         indicesBeforeAfter: Dict[int, int],
                         drawNumbers: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    srcImg = srcImg.copy()
    srcImg = cv2.cvtColor(srcImg, cv2.COLOR_GRAY2RGB)
    srcimgMarkovers = np.zeros_like(srcImg)
    dstImg = dstImg.copy()
    dstImg = cv2.cvtColor(dstImg, cv2.COLOR_GRAY2RGB)
    dstImgMarkovers = np.zeros_like(dstImg)

    # now fill matched particles with color
    numParticlesMatched = len(indicesBeforeAfter)
    if numParticlesMatched > 0:
        colorStep = 1 / numParticlesMatched
        hue = 0
        for origInd, targetInd in indicesBeforeAfter.items():
            color = colorsys.hsv_to_rgb(hue, 1, 1)
            color = color[0] * 255, color[1] * 255, color[2] * 255

            cnt = sourceContours[origInd]
            cv2.drawContours(srcimgMarkovers, [cnt], -1, color, -1)

            cnt = dstContours[targetInd]
            cv2.drawContours(dstImgMarkovers, [cnt], -1, color, -1)

            hue += colorStep

        srcImg = np.uint8(np.round(0.5 * srcImg + 0.5 * srcimgMarkovers))
        dstImg = np.uint8(np.round(0.5 * dstImg + 0.5 * dstImgMarkovers))

    # draw all contours with red outline, only for not matched particles
    for i, cnt in enumerate(sourceContours):
        if i not in indicesBeforeAfter.keys():
            cv2.drawContours(srcImg, [cnt], -1, (0, 0, 255), 2)

    for i, cnt in enumerate(dstContours):
        if i not in indicesBeforeAfter.values():
            cv2.drawContours(dstImg, [cnt], -1, (0, 0, 255), 2)

    if drawNumbers:
        fontSize = int(round(srcImg.shape[0] / 300))
        for origInd, targetInd in indicesBeforeAfter.items():
            x, y = int(round(sourceCenters[origInd, 0])), int(round(sourceCenters[origInd, 1]))
            cv2.putText(srcImg, str(origInd), (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), thickness=fontSize)

            x, y = int(round(dstCenters[targetInd, 0])), int(round(dstCenters[targetInd, 1]))
            cv2.putText(dstImg, str(origInd), (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), thickness=fontSize)

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot()
    # ax1.set_title(f'{len(indicesBeforeAfter)} of {sourceCenters.shape[0]} particles on {srcName} treatment image')
    # ax1.imshow(srcImg, cmap='gray')
    # fig1.tight_layout()
    # fig1.show()
    #
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot()
    # ax2.set_title(f'{len(indicesBeforeAfter)} of {dstCenters.shape[0]} particles on {dstName} treatment image')
    # ax2.imshow(dstImg, cmap='gray')
    # fig2.tight_layout()
    # fig2.show()
    # plt.show(block=True)
    return srcImg, dstImg


def getSnipImage(contour: np.ndarray, grayImg: np.ndarray, pxScale: float) -> np.ndarray:
    """
    Cuts out a snip image of the contour from the image. Also places a scalebar.
    :param contour: The particle's contour
    :param grayImg: Grayscale fullimage
    :param pxScale: Pixelscale in µm/px
    :return: snip image with scale bar
    """
    x, y, w, h = cv2.boundingRect(contour)  # we get x, y, width and height of the bounding box for the
    w = np.clip(w, 1, np.inf).astype(np.int)
    h = np.clip(h, 1, np.inf).astype(np.int)
    snipslice = grayImg[y:y + h, x:x + w].astype(np.uint8)  # extract the part of the image that is within bb + X px each way
    assert snipslice.shape[0] > 0 and snipslice.shape[1] > 0
    scalebar_length = 10  # length of scale bar in µm
    scalebar_width = round(scalebar_length / pxScale)  # width of scale bar in px
    if scalebar_width == 0:
        scalebar_width = 1
    scalebar_height = 5  # height of scale bar in px
    scalebar_x = round(snipslice.shape[1] * 0.1)  # find the x position to start the scale bar
    scalebar_y = round(snipslice.shape[0] * 0.9)  # find the y position to start the scale bar
    if scalebar_x + scalebar_width <= snipslice.shape[1]:  # test if scale bar would fit snip image width
        if snipslice[scalebar_y - scalebar_height:scalebar_y, scalebar_x:scalebar_x + scalebar_width].mean() <= 127:
            snipslice[scalebar_y - scalebar_height:scalebar_y, scalebar_x:scalebar_x + scalebar_width] = 255  # make white scale bar where background is dark
        else:  # make scale bar go across whole snip image if it does not fit
            snipslice[scalebar_y - scalebar_height:scalebar_y, scalebar_x:scalebar_x + scalebar_width] = 0  # make white scale bar where background is bright
    else:
        if snipslice[scalebar_y - scalebar_height:scalebar_y, scalebar_x:].mean() <= 127:
            snipslice[scalebar_y - scalebar_height:scalebar_y, scalebar_x:] = 255  # make white scale bar where background is dark
        else:
            snipslice[scalebar_y - scalebar_height:scalebar_y, scalebar_x:] = 0  # make white scale bar where background is

    return snipslice


def npImgArray_to_base64png(im):
    _, buffer = cv2.imencode('.png', np.ascontiguousarray(im))
    im64 = base64.b64encode(buffer).decode()
    im64formatted = 'data:image/png;base64,{}'.format(im64)
    return im64formatted
