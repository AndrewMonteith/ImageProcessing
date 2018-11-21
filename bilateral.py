'''
    Filename: bilaterial.py
    Purpose: Applies a bilateral filter onto an image
'''

import cv2
import numpy as np
from math import exp, sqrt, pi

# ----------------- / Constants
INPUT_IMAGE = 'test2.png'

# ------------------ / Filtering


def createBlank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def neighbourhood(x, y, width, height, kernelSize):
    kernelOffset = 0.5 * (kernelSize - 1)

    minX = int(max(0, x - kernelOffset))
    maxX = int(min(x + kernelOffset, width - 1))
    minY = int(max(y - kernelOffset, 0))
    maxY = int(min(y + kernelOffset, height - 1))

    for X in range(minX, maxX + 1):
        for Y in range(minY, maxY + 1):
            yield (X, Y)


def createGauss(sigma):
    k = 1 / (sigma * sqrt(2 * pi))
    K = 1 / ((-2) * sigma**2)

    return lambda x: k * exp(x**2 * K)


def bilaterial_filter(img, sigmaDistance, sigmaIntensity, kernelSize=5):

    sqrtTau = sqrt(2 * pi)

    def gaussDist(x):
        return (1 / (sigmaDistance * sqrtTau)) * exp(- (x**2) / (2 * sigmaDistance**2))

    def gaussIntensity(x):
        return (1 / (sigmaIntensity * sqrtTau)) * exp(- (x**2) / (2 * sigmaIntensity**2))

    (height, width, _) = img.shape

    def process_pixel(x, y):
        newB, newG, newR = 0, 0, 0
        normB, normG, normR = 0, 0, 0

        (origB, origG, origR) = img[y, x]
        intensities = [gaussIntensity(x) for x in range(0, 256)]

        for (nX, nY) in neighbourhood(x, y, width, height, kernelSize):
            (b, g, r) = map(int, img[nY, nX])

            # Computing Coefficients
            distCoef = gaussDist(sqrt((x - nX)**2 + (y - nY)**2))

            coefB = distCoef * intensities[abs(origB - b)]
            coefG = distCoef * intensities[abs(origG - g)]
            coefR = distCoef * intensities[abs(origR - r)]

            normB += coefB
            normG += coefG
            normR += coefR

            newB += (coefB * b)
            newG += (coefG * g)
            newR += (coefR * r)

        return (newB / normB, newG / normG, newR / normR)

    result = createBlank(width, height)

    for y in range(0, height):
        print(y, height)
        for x in range(0, width):
            result[y, x] = process_pixel(x, y)

    return result

    # ------------------ / Runner
if __name__ == "__main__":
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)

    if img is not None:
        sigmaIntensity = 20
        sigmaDistance = 18

        filtered = bilaterial_filter(
            img, sigmaDistance, sigmaIntensity, kernelSize=5)

        filtered_cv = cv2.bilateralFilter(
            img, 5, sigmaIntensity, sigmaDistance)

        cv2.imwrite('small_mine.png', filtered)
        cv2.imwrite('small_cv2.png', filtered_cv)
    else:
        print("Couldn't find image " + INPUT_IMAGE)
