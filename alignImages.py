from skimage import io
import cv2
import numpy as np
import scipy.optimize as sciOpt
import matplotlib.pyplot as plt
import time
from typing import List, Tuple
import detection_hysteresis as dh


def getContours(labelsImg: np.ndarray, minArea: float, maxArea: float) -> List[np.ndarray]:
    """
    Get Contours from labelled image
    :param labelsImg: the labelled image
    :param minArea: min Area of a particle to be considered (px**2)
    :param maxArea: max Area of a particle to be considered (px**2)
    :return: list of contours
    """
    selectedContours: List[np.ndarray] = []
    binImg = np.uint8(labelsImg > 0)
    contours, hierarchy = cv2.findContours(binImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for i, cnt in enumerate(contours):
        if minArea <= cv2.contourArea(cnt) <= maxArea and hierarchy[0, i, 3] < 0 and hierarchy[0, i, 2] < 0:
            selectedContours.append(cnt)
    return selectedContours


def getContourCenters(contourList: List[np.ndarray]) -> np.ndarray:
    """
    Takes a list fo contours and calculates their center coordinates
    :param contourList: the list of N contours
    :return: shape (N, 2) array of x, y coordinates of centers
    """
    centers: np.ndarray = np.zeros((len(contourList), 2))
    for index, contour in enumerate(contourList):
        centers[index, 0] = np.mean(contour[:, 0, 0])
        centers[index, 1] = np.mean(contour[:, 0, 1])
    return centers


def getContourCentersFromImage(image: np.ndarray, minArea: float, maxArea: float) -> np.ndarray:
    """
    Takes an image, finds particles (hysteresis detection) and returns their centers
    :param image: unlabelled color image
    :param minArea: min Area of a particle to be considered (px**2)
    :param maxArea: max Area of a particle to be considered (px**2)
    :return: shape (N, 2) array of x, y coordinates of centers
    """
    labelsImg, *others = dh.identify_particles(image)
    return getContourCenters(getContours(labelsImg, minArea, maxArea))


def offSetPoints(points: np.ndarray, angle: float, shift: np.ndarray) -> np.ndarray:
    """
    Rotates points around their mean and offsets them by given shift.
    :param points: shape (N, 2) array of points to recalculate
    :param angle: rotation angle
    :param shift: shape (2,) vector of shift
    """
    sin, cos = np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))
    pointsMean: np.ndarray = np.mean(points, axis=0)
    pointsCentered: np.ndarray = points - pointsMean
    pointsRot: np.ndarray = np.zeros_like(points, dtype=np.float)
    for i in range(points.shape[0]):
        pointsRot[i, 0] = pointsCentered[i, 0]*cos - pointsCentered[i, 1]*sin + pointsMean[0]
        pointsRot[i, 1] = pointsCentered[i, 0]*sin + pointsCentered[i, 1]*cos + pointsMean[1]

    return pointsRot + shift


def getErrorFromCenters(lessPoints: np.ndarray, morePoints: np.ndarray) -> Tuple[float, List[int]]:
    """
    Calculates the distance of all centers and report the assignment of points in centers2 to points in center1
    :param lessPoints: (N x 2) shape array of x, y coordinates
    :param morePoints: (M x 2) shape array of x, y coordinates, ideally M >= N
    :return: tuple: error (float), list of indices mapping the shorter list of points to the longer list of points
    """
    if morePoints.shape[0] < lessPoints.shape[0]:
        morePoints, lessPoints = lessPoints, morePoints

    err: float = 0.0
    copyMorePoints: np.ndarray = morePoints.copy()
    indices: list[int] = []
    i: int = 0
    while i < lessPoints.shape[0]:
        curPoint = lessPoints[i]
        distances: np.ndarray = np.linalg.norm(copyMorePoints - curPoint, axis=1)
        closestPointIndex = np.argmin(distances)
        err += distances[closestPointIndex]
        
        closestPoint: np.ndarray = copyMorePoints[closestPointIndex, :]
        distances = np.linalg.norm(morePoints - closestPoint, axis=1)
        indices.append(int(np.argmin(distances)))
        assert distances[indices[-1]] == 0
        copyMorePoints = np.delete(copyMorePoints, closestPointIndex, axis=0)
        
        i += 1

    return err, indices


def getDiffOfAngleShift(angleShift: np.ndarray, origPoints: np.ndarray, knownDstPoints: np.ndarray) -> np.ndarray:
    """
    Applies angle and shift to the origPoints and calulates differences to the knownDstPoints.
    First, the transform is applied to origPoints. Then, the transformed Points are mapped to the known Dst points
    to get the lowerst possible error.
    :param angleShift: shape(3) array, [0] = angle (degree), [1, 2] = x, y offset
    :param origPoints: Nx2 array of N source points
    :param knownDstPoints: Mx2 array of M target points, M can or cannot be equal N
    :return: ravelled differences of all coordinates, suitable for least_square optimization
    """
    origPoints = origPoints.astype(np.float)  # just to make sure not to have any integer datatypes...
    knownDstPoints = knownDstPoints.astype(np.float)
    transformedPoints = offSetPoints(origPoints, angleShift[0], np.array([angleShift[1], angleShift[2]]))

    lowerNumPoints: int = min(transformedPoints.shape[0], knownDstPoints.shape[0])
    srcPoints: np.ndarray = np.zeros((lowerNumPoints, 2), dtype=np.float)
    dstPoints: np.ndarray = np.zeros((lowerNumPoints, 2), dtype=np.float)

    err, ind = getErrorFromCenters(transformedPoints, knownDstPoints)
    assert len(ind) == lowerNumPoints

    if transformedPoints.shape[0] <= knownDstPoints.shape[0]:
        srcPoints = transformedPoints
        for origInd, associatedInd in enumerate(ind):
            dstPoints[origInd, :] = knownDstPoints[associatedInd]
    else:
        dstPoints = knownDstPoints
        for origInd, associatedInd in enumerate(ind):
            srcPoints[origInd] = transformedPoints[associatedInd]

    err: np.ndarray = (srcPoints - dstPoints).ravel()

    return err


def findAngleAndShift(points1: np.ndarray, points2: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Finds the best fitting angle and shift for the given pair of points. They don't have to be ordered.
    :return tuple(optAngle: float, optShift: (1x2)np.ndarray)
    """
    points1 = points1.astype(np.float)
    points2 = points2.astype(np.float)
    errFunc = lambda x: getDiffOfAngleShift(x, points1, points2)
    xstart = np.array([0, 0, 0])
    opt = sciOpt.least_squares(errFunc, xstart, bounds=(np.array([-45, -10, -10]), np.array([45, 10, 10])),
                               method='dogbox')
    optAngle, optShift = opt.x[0], opt.x[1:]
    return optAngle, optShift


if __name__ == '__main__':

    t0 = time.time()
    srcImg: np.ndarray = io.imread(r'w02a_pre_30percent.tif')
    dstImg: np.ndarray = io.imread(r'w02a_water_30percent.tif')

    srcImg = cv2.medianBlur(srcImg, ksize=9)
    dstImg = cv2.medianBlur(dstImg, ksize=9)

    minParticleArea: float = 500
    maxParticleArea: float = 1e6

    sourceCenters: np.ndarray = getContourCentersFromImage(srcImg, minParticleArea, maxParticleArea)
    dstCenters: np.ndarray = getContourCentersFromImage(dstImg, minParticleArea, maxParticleArea)
    print(f'numSourcePoints {sourceCenters.shape[0]}, numDstPoints {dstCenters.shape[0]}')
    print(f'loading images and getting contours took {time.time()-t0} seconds')

    t0 = time.time()
    angle, shift = findAngleAndShift(sourceCenters, dstCenters)
    print(angle, shift)
    print(f'getting transform and error took {time.time()-t0} seconds')

    transformed: np.ndarray = offSetPoints(sourceCenters, angle, shift)
    error, indices = getErrorFromCenters(transformed, dstCenters)

    inverted: bool = False
    if sourceCenters.shape[0] > dstCenters.shape[0]:
        inverted = True

    for origInd, targetInd in enumerate(indices):
        if not inverted:
            x, y = int(round(sourceCenters[origInd, 0])), int(round(sourceCenters[origInd, 1]))
        else:
            x, y = int(round(sourceCenters[targetInd, 0])), int(round(sourceCenters[targetInd, 1]))

        cv2.putText(srcImg, str(origInd), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 255, 255), thickness=5)

        if not inverted:
            x, y = int(round(dstCenters[targetInd, 0])), int(round(dstCenters[targetInd, 1]))
        else:
            x, y = int(round(dstCenters[origInd, 0])), int(round(dstCenters[origInd, 1]))

        cv2.putText(dstImg, str(origInd), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 255, 255), thickness=5)

    # display results
    plt.subplot(121)
    plt.title(f'{len(indices)} particles on w02a_pre')
    plt.imshow(srcImg, cmap='gray')
    plt.scatter(sourceCenters[:, 0], sourceCenters[:, 1], alpha=0.4)

    plt.subplot(122)
    plt.title(f'{len(indices)} particles on w02a_water')
    plt.imshow(dstImg, cmap='gray')
    plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.4)

    plt.show()
