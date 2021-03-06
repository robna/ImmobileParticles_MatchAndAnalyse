import unittest
import numpy as np
import alignImages as ai


class TestImageAlign(unittest.TestCase):
    def assert_array_almost_equal(self, arr1: np.ndarray, arr2: np.ndarray, places: int = 7) -> None:
        self.assertEqual(arr1.shape, arr2.shape)
        arr1, arr2 = arr1.ravel(), arr2.ravel()
        for elem1, elem2 in zip(arr1, arr2):
            self.assertAlmostEqual(elem1, elem2, places=places)

    def test_get_error_from_centers(self):
        points: np.ndarray = np.arange(10).reshape((5, 2))
        errors, indBefore2After = ai.getIndicesAndErrosFromCenters(points, points)
        self.assertEqual(np.sum(errors), 0.0)
        for key, val in indBefore2After.items():
            self.assertEqual(key, val)

        # offset points by a bit
        points2: np.ndarray = points + np.array([0.01, 0])
        errors, indBefore2After = ai.getIndicesAndErrosFromCenters(points, points2)

        for key, val in indBefore2After.items():
            self.assertEqual(key, val)
        self.assertAlmostEqual(np.sum(errors), 0.05)

        pointsCopy = points.copy()
        # move out first point too far
        pointsCopy[0, :] += np.array([5, 0])
        # also, add a list of max Errors per particle that are being used as distance tolerance for the matching:
        maxErrors = [5, 0, 0, 0, 0]  # first point is allowed to be max 5 units away
        errors, indBefore2After = ai.getIndicesAndErrosFromCenters(points, pointsCopy, maxErrors)
        self.assertEqual(np.sum(errors), 5)  # Result is fine

        maxErrors = [0, 5, 0, 0, 0]  # second point is allowed to be max 5 units away, but not the first
        errors, indBefore2After = ai.getIndicesAndErrosFromCenters(points, pointsCopy, maxErrors)
        self.assertEqual(np.sum(errors), 100000)  # the 100000 is added as high penalty, as the first point is too far away
        self.assertEqual(len(indBefore2After), 4)  # and only four particles are matched
        self.assertTrue(0 not in indBefore2After.keys())  # and particle 0 is not in the dictionary

        maxErrors = [4.9, 0, 0, 0, 0]  # Tolerance for first point just below actual distance
        errors, indBefore2After = ai.getIndicesAndErrosFromCenters(points, pointsCopy, maxErrors)
        self.assertEqual(np.sum(errors), 100000)  # the 100000 is added as high penalty, as the first point is too far away
        self.assertEqual(len(indBefore2After), 4)  # and only four particles are matched
        self.assertTrue(0 not in indBefore2After.keys())  # and particle 0 is not in the dictionary

        points2Copy = points2.copy()
        # let's swap some of the points
        points2[0, :], points2[1, :] = points2Copy[1, :], points2Copy[0, :]
        points2[2, :], points2[4, :] = points2Copy[4, :], points2Copy[2, :]
        errors, indBefore2After = ai.getIndicesAndErrosFromCenters(points, points2)
        expectedInd = {0: 1,
                       1: 0,
                       2: 4,
                       3: 3,
                       4: 2}

        self.assertAlmostEqual(np.sum(errors), 0.05)
        self.assertEqual(len(indBefore2After), 5)
        for key in indBefore2After.keys():
            self.assertEqual(indBefore2After[key], expectedInd[key])

        # lets add distance tolerance to that...
        maxErrors = [0, 0.1, 0.1, 0.1, 0.1]  # first point should be thrown out as it only allows for 0 error.
        errors, indBefore2After = ai.getIndicesAndErrosFromCenters(points, points2, maxErrors)
        self.assertAlmostEqual(np.sum(errors), 100000.04)
        self.assertTrue(0 not in indBefore2After.keys())


        # add points that are outliers
        points3: np.ndarray = np.zeros((7, 2))
        points3[:5, :] = points2
        points3[5:, :] = np.array([[10, 20], [-2.5, 3.5]])
        errors, indBefore2After = ai.getIndicesAndErrosFromCenters(points, points3)
        self.assertAlmostEqual(np.sum(errors), 0.05)
        for key in indBefore2After.keys():
            self.assertEqual(indBefore2After[key], expectedInd[key])

        points4: np.ndarray = points2[:3, :]  # let's have less points in the second array
        errors, indBefore2After = ai.getIndicesAndErrosFromCenters(points, points4)
        self.assertAlmostEqual(np.sum(errors), 0.03)
        expectedInd = {1: 0,
                       0: 1,
                       4: 2}
        for key in indBefore2After.keys():
            self.assertEqual(indBefore2After[key], expectedInd[key])

    def test_offset_points(self):
        points: np.ndarray = np.arange(10).reshape((5, 2))
        # only offsets
        for shiftX in [-5, 0, 5]:
            for shiftY in [-3, 0, 3]:
                shift: np.ndarray = np.array([shiftX, shiftY])
                newPoints: np.ndarray = ai.offSetPoints(points, 0, shift)
                for i in range(points.shape[0]):
                    self.assertEqual(points[i, 0] + shift[0], newPoints[i, 0])
                    self.assertEqual(points[i, 1] + shift[1], newPoints[i, 1])

        # only angles:
        shift = np.array([0, 0])
        points = np.array([[0, 1], [0, -1]])
        angle = 90
        newPoints = ai.offSetPoints(points, angle, shift)
        self.assert_array_almost_equal(newPoints, np.array([[-1, 0], [1, 0]]))

        angle = 180
        newPoints = ai.offSetPoints(points, angle, shift)
        self.assert_array_almost_equal(newPoints, np.array([[0, -1], [0, 1]]))

        angle = 90
        points = np.array([[1, 1], [-1, -1]])
        newPoints = ai.offSetPoints(points, angle, shift)
        self.assert_array_almost_equal(newPoints, np.array([[-1, 1],  [1, -1]]))

        # rot and offset:
        shift = [2, 1]
        newPoints = ai.offSetPoints(points, angle, shift)
        self.assert_array_almost_equal(newPoints, np.array([[1, 2], [3, 0]]))

    def test_get_offset_and_rot(self):
        points: np.ndarray = np.array([[1, 1], [1, 0], [0, 2], [1, 2]])
        shifts: list = [[0, 0], [1, 0], [1, 1], [-1, -1]]
        angles: list = [-20, -10, 0, 10, 20]
        for shift in shifts:
            for angle in angles:
                newPoints: np.ndarray = ai.offSetPoints(points, angle, np.array(shift))
                optAngle, optShift = ai.findAngleAndShift(points, newPoints)
                self.assertAlmostEqual(optAngle, angle, places=3)
                self.assertAlmostEqual(optShift[0], shift[0], places=5)
                self.assertAlmostEqual(optShift[1], shift[1], places=5)

                transformedPoints: np.ndarray = ai.offSetPoints(points, optAngle, optShift)
                err, _ = ai.getIndicesAndErrosFromCenters(newPoints, transformedPoints)
                self.assertAlmostEqual(np.sum(err), 0, places=3)
                self.assert_array_almost_equal(transformedPoints, newPoints)

                newPoints2 = newPoints.copy()
                newPoints2[2, :], newPoints2[0, :] = newPoints[0, :], newPoints[2, :]  # switch indices
                optAngle, optShift = ai.findAngleAndShift(points, newPoints2)
                transformedPoints = ai.offSetPoints(points, optAngle, optShift)
                err, _ = ai.getIndicesAndErrosFromCenters(transformedPoints, newPoints2)
                self.assertAlmostEqual(np.sum(err), 0, places=6)
                self.assertAlmostEqual(optAngle, angle, places=6)
                self.assertAlmostEqual(optShift[0], shift[0])
                self.assertAlmostEqual(optShift[1], shift[1])

                newPoints3 = np.zeros((newPoints.shape[0] + 2, 2))
                newPoints3[:newPoints2.shape[0], :] = newPoints2
                newPoints3[newPoints2.shape[0]:, :] = np.array([[-100, 100], [100, 100]])
                optAngle, optShift = ai.findAngleAndShift(points, newPoints3)
                transformedPoints: np.ndarray = ai.offSetPoints(points, optAngle, optShift)
                err, _ = ai.getIndicesAndErrosFromCenters(transformedPoints, newPoints3)
                self.assertAlmostEqual(np.sum(err), 0, places=6)
                self.assertAlmostEqual(optAngle, angle, places=6)
                self.assertAlmostEqual(optShift[0], shift[0])
                self.assertAlmostEqual(optShift[1], shift[1])


if __name__ == '__main__':
    unittest.main()
