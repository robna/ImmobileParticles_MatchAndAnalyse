import cv2
import numpy as np
import scipy.optimize as sciOpt
from scipy.spatial import distance_matrix
from typing import List, Tuple, Dict
import detection_hysteresis as dh


def getContours(labelsImg: np.ndarray, minArea: float, maxArea: float) -> Tuple[List[np.ndarray], List[int]]:
    """
    Get Contours from labelled image
    :param labelsImg: the labelled image
    :param minArea: min Area of a particle to be considered (px**2)
    :param maxArea: max Area of a particle to be considered (px**2)
    :return: list of contours
    """
    selectedContours: List[np.ndarray] = []
    binImg = np.uint8(labelsImg > 0)
    skipIndices: List[int] = []
    contours, hierarchy = cv2.findContours(binImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    for i, cnt in enumerate(contours):
        if minArea <= cv2.contourArea(cnt) <= maxArea and hierarchy[0, i, 3] < 0 and hierarchy[0, i, 2] < 0:
            selectedContours.append(cnt)
        else:
            skipIndices.append(i)
    return selectedContours, skipIndices


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


def getLabelsAndContoursFromImage(image: np.ndarray, params: dh.RecognitionParameters = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Takes an image, finds particles (hysteresis detection) and returns their contours
    :param image: unlabelled color image
    :param params: Recognition Parameters
    :return: List of contours
    """
    if params is None:
        params = dh.RecognitionParameters()

    labelsImg, *others = dh.identify_particles(image, params)
    contours, skipIndices = getContours(labelsImg, params.minArea, params.maxArea)
    for i in skipIndices:
        labelsImg[labelsImg == i+1] = 0
    return labelsImg, contours


def getContourCentersFromImage(image: np.ndarray, params: dh.RecognitionParameters = dh.RecognitionParameters()) -> np.ndarray:
    """
    Takes an image, finds particles (hysteresis detection) and returns their centers
    :param image: unlabelled color image
    :param params: Recognition Parameters
    :return: shape (N, 2) array of x, y coordinates of centers
    """
    labelsImg, *others = dh.identify_particles(image, params)
    return getContourCenters(getContours(labelsImg, params.minArea, params.maxArea))


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


def getIndicesAndErrosFromCenters(beforePoints: np.ndarray, afterPoints: np.ndarray,
                                  maxDistError: float = np.inf) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Calculates the distance of all centers and report the assignment of points in lessPoints to points in morePoints
    :param beforePoints: (N x 2) shape array of x, y coordinates
    :param afterPoints: (M x 2) shape array of x, y coordinates, ideally M >= N
    :param maxDistError: maximum tolerated distance between points to be accepted as pairings
    :return: tuple: error (float), list of indices mapping particleIndices from before image to after image
    """
    inverted: bool = False
    morePoints, lessPoints = beforePoints, afterPoints
    if morePoints.shape[0] < lessPoints.shape[0]:
        inverted = True
        morePoints, lessPoints = lessPoints, morePoints

    errors: List[float] = []
    indicesBefore2After: Dict[int, int] = {}
    distMat: np.ndarray = distance_matrix(lessPoints, morePoints)

    for _ in range(lessPoints.shape[0]):
        i, j = np.unravel_index(distMat.argmin(), distMat.shape)  # get the pairing which is closest together
        minDist = distMat[i, j]
        distMat[i, :] = np.inf  # effectively removes these indices from further consideration
        distMat[:, j] = np.inf  # effectively removes these indices from further consideration

        if minDist <= maxDistError:
            if inverted:
                indicesBefore2After[i] = j
            else:
                indicesBefore2After[j] = i
            errors.append(minDist)
        else:
            errors.append(100000)  # add a high penalty

    return np.array(errors), indicesBefore2After


def getDiffOfAngleShift(angleShift: np.ndarray, origPoints: np.ndarray, knownDstPoints: np.ndarray,
                        maxDistError: float = np.inf) -> np.ndarray:
    """
    Applies angle and shift to the origPoints and calulates differences to the knownDstPoints.
    First, the transform is applied to origPoints. Then, the transformed Points are mapped to the known Dst points
    to get the lowerst possible error.
    :param angleShift: shape(3) array, [0] = angle (degree), [1, 2] = x, y offset
    :param origPoints: Nx2 array of N source points
    :param knownDstPoints: Mx2 array of M target points, M can or cannot be equal N
    :param maxDistError: maximum tolerated distance between points to be accepted as pairings
    :return: ravelled differences of all coordinates, suitable for least_square optimization
    """
    origPoints = origPoints.astype(np.float)  # just to make sure not to have any integer datatypes...
    knownDstPoints = knownDstPoints.astype(np.float)
    transformedPoints = offSetPoints(origPoints, angleShift[0], np.array([angleShift[1], angleShift[2]]))

    err, ind = getIndicesAndErrosFromCenters(transformedPoints, knownDstPoints, maxDistError)
    assert len(ind) > 0
    return err


def findAngleAndShift(points1: np.ndarray, points2: np.ndarray,
                      maxDistError: float = np.inf) -> Tuple[float, np.ndarray]:
    """
    Finds the best fitting angle and shift for the given pair of points. They don't have to be ordered.
    :return tuple(optAngle: float, optShift: (1x2)np.ndarray)
    """
    points1 = points1.astype(np.float)
    points2 = points2.astype(np.float)
    errFunc = lambda x: getDiffOfAngleShift(x, points1, points2, maxDistError)
    xstart = np.array([0, 0, 0])
    opt = sciOpt.least_squares(errFunc, xstart, bounds=(np.array([-45, -np.inf, -np.inf]), np.array([45, np.inf, np.inf])),
                               method='dogbox')
    optAngle, optShift = opt.x[0], opt.x[1:]
    return optAngle, optShift


def getCentralPoints(points: np.ndarray) -> np.ndarray:
    xmin, ymin = points[:, 0].min(), points[:, 1].min()
    xmax, ymax = points[:, 0].max(), points[:, 1].max()
    diffX, diffY = xmax - xmin, ymax - ymin
    centerx, centery = xmin + diffX / 2, ymin + diffY / 2

    lowX, highX = centerx - diffX/3, centerx + diffX/3
    lowY, highY = centery - diffY/3, centery + diffY/3

    newPoints: List[np.ndarray] = []
    for point in points:
        if lowX <= point[0] <= highX and lowY <= point[1] <= highY:
            newPoints.append(point)

    return np.array(newPoints)
