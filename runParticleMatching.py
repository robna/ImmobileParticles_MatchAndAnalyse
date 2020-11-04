from skimage import io
import matplotlib.pyplot as plt
import time
import os
from alignImages import *
from detection_hysteresis import getLowerThresholdForImg, RecognitionParameters

t0 = time.time()

# pathSrcImg = r'w02a_pre_30percent.tif'
# pathDstImg = r'w02a_water_30percent.tif'

pathSrcImg = r'w20_pre_30percent.tif'
pathDstImg = r'w20_HCl_30percent.tif'

srcImg: np.ndarray = io.imread(pathSrcImg)
dstImg: np.ndarray = io.imread(pathDstImg)

srcImg = cv2.medianBlur(srcImg, ksize=9)
dstImg = cv2.medianBlur(dstImg, ksize=9)

params: RecognitionParameters = RecognitionParameters()
params.highTreshold = 0.6
params.lowerThreshold = getLowerThresholdForImg(srcImg) * 1.5
params.minArea = 100
params.maxArea = 1500
sourceContours: List[np.ndarray] = getContoursFromImage(srcImg, params)
sourceCenters: np.ndarray = getContourCenters(sourceContours)

dstContours: List[np.ndarray] = getContoursFromImage(dstImg, params)
dstCenters: np.ndarray = getContourCenters(dstContours)
print(f'numSourcePoints {sourceCenters.shape[0]}, numDstPoints {dstCenters.shape[0]}')
print(f'loading images and getting contours took {time.time()-t0} seconds')

t0 = time.time()
angle, shift = findAngleAndShift(sourceCenters, dstCenters)
print(angle, shift)
print(f'getting transform and error took {time.time()-t0} seconds')

transformed: np.ndarray = offSetPoints(sourceCenters, angle, shift)
error, indices = getIndicesAndErrosFromCenters(transformed, dstCenters)

inverted: bool = False
if sourceCenters.shape[0] > dstCenters.shape[0]:
    inverted = True

for origInd, targetInd in enumerate(indices):
    if not inverted:
        x, y = int(round(sourceCenters[origInd, 0])), int(round(sourceCenters[origInd, 1]))
    else:
        x, y = int(round(sourceCenters[targetInd, 0])), int(round(sourceCenters[targetInd, 1]))

    cv2.putText(srcImg, str(origInd), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5.0, 255, thickness=5)

    if not inverted:
        x, y = int(round(dstCenters[targetInd, 0])), int(round(dstCenters[targetInd, 1]))
    else:
        x, y = int(round(dstCenters[origInd, 0])), int(round(dstCenters[origInd, 1]))

    cv2.putText(dstImg, str(origInd), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5.0, 255, thickness=5)

cv2.drawContours(srcImg, sourceContours, -1, 255, 2)
cv2.drawContours(dstImg, dstContours, -1, 255, 2)

# display results
fig1 = plt.figure()
ax1 = fig1.add_subplot()
srcName = os.path.basename(pathSrcImg).split('tif')[0]
ax1.set_title(f'{len(indices)} of {sourceCenters.shape[0]} particles on {srcName}')
ax1.imshow(srcImg, cmap='gray')
# ax1.scatter(sourceCenters[:, 0], sourceCenters[:, 1], color='green', alpha=0.2)
fig1.tight_layout()
fig1.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot()
dstName = os.path.basename(pathDstImg).split('tif')[0]
ax2.set_title(f'{len(indices)} of {dstCenters.shape[0]} particles on {dstName}')
ax2.imshow(dstImg, cmap='gray')
# ax2.scatter(transformed[:, 0], transformed[:, 1], color='green', alpha=0.2)
fig2.tight_layout()
fig2.show()
