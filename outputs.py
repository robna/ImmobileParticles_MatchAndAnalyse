from typing import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
if TYPE_CHECKING:
    import pandas as pd


def generateOutputGraphs(sourceCenters: np.ndarray, dstCenters: np.ndarray,
                         sourceContours: List[np.ndarray], dstContours: List[np.ndarray],
                         srcImg: np.ndarray, dstImg: np.ndarray,
                         srcName: str, dstName: str,
                         indicesBeforeAfter: Dict[int, int]) -> Tuple[plt.figure, plt.figure]:

    for origInd, targetInd in indicesBeforeAfter.items():
        x, y = int(round(sourceCenters[origInd, 0])), int(round(sourceCenters[origInd, 1]))
        cnt = sourceContours[origInd]

        cv2.drawContours(srcImg, [cnt], -1, 255, 2)
        cv2.putText(srcImg, str(origInd), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5.0, 255, thickness=5)

        x, y = int(round(dstCenters[targetInd, 0])), int(round(dstCenters[targetInd, 1]))
        cnt = dstContours[targetInd]

        cv2.drawContours(dstImg, [cnt], -1, 255, 2)
        cv2.putText(dstImg, str(origInd), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5.0, 255, thickness=5)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_title(f'{len(indicesBeforeAfter)} of {sourceCenters.shape[0]} particles on {srcName}')
    ax1.imshow(srcImg, cmap='gray')
    fig1.tight_layout()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_title(f'{len(indicesBeforeAfter)} of {dstCenters.shape[0]} particles on {dstName}')
    ax2.imshow(dstImg, cmap='gray')
    fig2.tight_layout()
    return fig1, fig2


def getRatioOfProperty(prop: str, dfBefore: 'pd.DataFrame', dfAfter: 'pd.DataFrame', indicesBeforeAfter: Dict[int, int]) -> np.ndarray:
    ratios: np.ndarray = np.zeros((len(indicesBeforeAfter), ))  # property before / property after for each index

    assert prop in dfBefore.columns and prop in dfAfter.columns, f'Requested property {prop} not found in dataframe'
    propBefore: pd.Series = dfBefore.get(prop)
    propAfter: pd.Series = dfAfter.get(prop)

    for i, (origInd, targetInd) in enumerate(indicesBeforeAfter.items()):
        ratios[i] = propBefore[origInd] / propAfter[targetInd]

    return ratios
