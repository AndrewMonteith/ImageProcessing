'''
    Filename: bilaterial.py
    Purpose: Applies a bilateral filter onto an image
'''

import cv2
import numpy as np
from math import exp, sqrt, pi, e

# ----------------- / Constants
INPUT_NON_FLASH = 'test3a.jpg'
INPUT_FLASH = 'test3b.jpg'
FILTERED_IMAGE_NAME = 'result.jpg'

# ------------------ / Filtering


def createBlankImage(width, height, channels):
    """Create new image(numpy array)"""
    return np.zeros((height, width, channels), np.uint8)


def neighbourhood(x, y, width, height, kernelSize):
    "Generator that returns all neighbourhood coordinates for a given pixel"
    kernelOffset = 0.5 * (kernelSize - 1)

    minX = int(max(0, x - kernelOffset))
    maxX = int(min(x + kernelOffset, width - 1))
    minY = int(max(y - kernelOffset, 0))
    maxY = int(min(y + kernelOffset, height - 1))

    for X in range(minX, maxX + 1):
        for Y in range(minY, maxY + 1):
            yield (X, Y)


def createGauss(sigma):
    "Constructs a Gaussian Function for a given sigma"
    k = 1 / (sigma * sqrt(2 * pi))
    K = 1 / ((-2) * sigma**2)

    return lambda x: k * e ** (x**2 * K)


def joint_bilateral_filter(imgNonFlash, imgFlash, sigmaDistance, sigmaIntensity, kernelSize=5):
    "Applies Joint Bilateral Filter to image pair (imageNonFlash, imageFlash)"

    gaussDist = createGauss(sigmaDistance)
    gaussIntensity = createGauss(sigmaIntensity)

    (height, width, channels) = imgNonFlash.shape

    intensityCoefficients = [gaussIntensity(x) for x in range(0, 256)]

    def process_pixel(x, y):
        "Computes new colour vector for each pixel (x, y)"

        newColourVector = [0] * channels
        normalisationFactors = [0] * channels
        coefficients = [0] * channels

        centerFlashVector = imgFlash[y, x]

        for (nX, nY) in neighbourhood(x, y, width, height, kernelSize):
            neighbourNonFlashVector = imgNonFlash[nY, nX]
            neighbourFlashVector = imgFlash[nY, nX]

            distanceCoefficient = gaussDist(sqrt((x - nX)**2 + (y - nY)**2))

            for i in range(0, channels):
                coefficient = distanceCoefficient * intensityCoefficients[abs(neighbourFlashVector[i] - centerFlashVector[i])]
                coefficients[i] = coefficient
                normalisationFactors[i] += coefficient
                newColourVector[i] += coefficient * neighbourNonFlashVector[i]

        # Normalise Each Colour.
        for i in range(0, channels):
            newColourVector[i] = newColourVector[i] / normalisationFactors[i]

        return newColourVector

    result = createBlankImage(width, height, channels)

    for y in range(0, height):
        for x in range(0, width):
            result[y, x] = process_pixel(x, y)

    return result

    # ------------------ / Runner
if __name__ == "__main__":
    imgFlash = cv2.imread(INPUT_FLASH, cv2.IMREAD_UNCHANGED)
    imgNoFlash = cv2.imread(INPUT_NON_FLASH, cv2.IMREAD_UNCHANGED)

    if (imgFlash is not None) and (imgNoFlash is not None):
        sigmaIntensity = 20
        sigmaDistance = 1.8
        kernelSize = 11

        filtered = joint_bilateral_filter(
            imgNoFlash, imgFlash, sigmaDistance, sigmaIntensity, kernelSize)

        cv2.imwrite(FILTERED_IMAGE_NAME, filtered)
    else:
        print("Couldn't find images")
