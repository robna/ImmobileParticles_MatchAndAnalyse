from skimage import io
import matplotlib.pyplot as plt
import time
import os
from alignImages import *
from detection_hysteresis import getLowerThresholdForImg, RecognitionParameters

t0 = time.time()

# pathSrcImg = r'w02a_pre_30percent.tif'
# pathDstImg = r'w02a_water_30percent.tif'

# pathSrcImg = r'w20_pre_30percent.tif'
# pathDstImg = r'w20_HCl_30percent.tif'

# pathSrcImg = r'w20_pre.tif'
# pathDstImg = r'w20_HCl.tif'

pathSrcImg = r'w05_pre.tif'
pathDstImg = r'w05_KOH.tif'

srcImg: np.ndarray = io.imread(pathSrcImg)
dstImg: np.ndarray = io.imread(pathDstImg)

scaleFac: float = 0.3
srcImg = cv2.resize(srcImg, None, fx=scaleFac, fy=scaleFac)
dstImg = cv2.resize(dstImg, None, fx=scaleFac, fy=scaleFac)

srcImg = cv2.medianBlur(srcImg, ksize=9)
dstImg = cv2.medianBlur(dstImg, ksize=9)

params: RecognitionParameters = RecognitionParameters()
params.highTreshold = 0.6
params.lowerThreshold = getLowerThresholdForImg(srcImg) * 1.5
params.minArea = 100
params.maxArea = 15000
params.doHysteresis = False  # Don't do hysteresis first, just to get all particles (better for getting transform)

sourceContours: List[np.ndarray] = getContoursFromImage(srcImg, params)
sourceCenters: np.ndarray = getContourCenters(sourceContours)

dstContours: List[np.ndarray] = getContoursFromImage(dstImg, params)
dstCenters: np.ndarray = getContourCenters(dstContours)
print(f'loading images and getting contours took {time.time()-t0} seconds')

t0 = time.time()
ymin, ymax = sourceCenters[:, 1].min(), sourceCenters[:, 1].max()
maxDistError = (ymax - ymin) * 0.02  # i.e., 2 percent of maximum y range (just to kind of map relative to img resolution)
angle, shift = findAngleAndShift(sourceCenters, dstCenters, maxDistError)
print(f'angle: {angle}, shift: {shift}, numSrcCenters: {len(sourceContours)}, numDstCenters: {len(dstCenters)}')
print(f'getting transform and error took {time.time()-t0} seconds')

t0 = time.time()
# No get only the relevant particles
params.doHysteresis = True
sourceContours = getContoursFromImage(srcImg, params)
sourceCenters = getContourCenters(sourceContours)
dstContours = getContoursFromImage(dstImg, params)
dstCenters = getContourCenters(dstContours)

transformed: np.ndarray = offSetPoints(sourceCenters, angle, shift)
error, indices = getIndicesAndErrosFromCenters(transformed, dstCenters, maxDistError)

# Display results
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
