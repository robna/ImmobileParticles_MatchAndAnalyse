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
                         indices: Dict[int, int]) -> Tuple[plt.figure, plt.figure]:

    inverted: bool = False
    if sourceCenters.shape[0] > dstCenters.shape[0]:
        inverted = True

    for origInd, targetInd in indices.items():
        if not inverted:
            x, y = int(round(sourceCenters[origInd, 0])), int(round(sourceCenters[origInd, 1]))
            cnt = sourceContours[origInd]
        else:
            x, y = int(round(sourceCenters[targetInd, 0])), int(round(sourceCenters[targetInd, 1]))
            cnt = sourceContours[targetInd]

        cv2.drawContours(srcImg, [cnt], -1, 255, 2)
        cv2.putText(srcImg, str(origInd), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5.0, 255, thickness=5)

        if not inverted:
            x, y = int(round(dstCenters[targetInd, 0])), int(round(dstCenters[targetInd, 1]))
            cnt = dstContours[targetInd]
        else:
            x, y = int(round(dstCenters[origInd, 0])), int(round(dstCenters[origInd, 1]))
            cnt = dstContours[origInd]

        cv2.drawContours(dstImg, [cnt], -1, 255, 2)
        cv2.putText(dstImg, str(origInd), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5.0, 255, thickness=5)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_title(f'{len(indices)} of {sourceCenters.shape[0]} particles on {srcName}')
    ax1.imshow(srcImg, cmap='gray')
    # ax1.scatter(sourceCenters[:, 0], sourceCenters[:, 1], color='green', alpha=0.2)
    fig1.tight_layout()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_title(f'{len(indices)} of {dstCenters.shape[0]} particles on {dstName}')
    ax2.imshow(dstImg, cmap='gray')
    # ax2.scatter(transformed[:, 0], transformed[:, 1], color='green', alpha=0.2)
    fig2.tight_layout()
    return fig1, fig2


def getRatioOfProperty(prop: str, dfBefore: 'pd.DataFrame', dfAfter: 'pd.DataFrame', indices: Dict[int, int]) -> np.ndarray:
    inverted: bool = len(dfBefore) > len(dfAfter)
    ratios: np.ndarray = np.zeros((len(indices), ))  # property before / property after for each index

    assert prop in dfBefore.columns and prop in dfAfter.columns, f'Requested property {prop} not found in dataframe'
    propBefore: pd.Series = dfBefore.get(prop)
    propAfter: pd.Series = dfAfter.get(prop)

    for i, (origInd, targetInd) in enumerate(indices.items()):
        if not inverted:
            ratios[i] = propBefore[origInd] / propAfter[targetInd]
        else:
            ratios[i] = propBefore[targetInd] / propAfter[origInd]

    return ratios
