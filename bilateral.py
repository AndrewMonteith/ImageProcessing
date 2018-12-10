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

    return lambda x: k * e ** (x**2 * K)


def joint_bilateral_filter(imgNonFlash, imgFlash, sigmaDistance, sigmaIntensity, kernelSize=5):
    gaussDist = createGauss(sigmaDistance)
    gaussIntensity = createGauss(sigmaIntensity)

    (height, width, channels) = imgNonFlash.shape

    intensityCoefficients = [gaussIntensity(x) for x in range(0, 256)]
    
    def process_pixel(x, y):
        newColours = [0]*channels
        normalisationFactors = [0]*channels

        centerFlashVector = imgFlash[y, x]

        coefficients = [0] * channels
        for (nX, nY) in neighbourhood(x, y, width, height, kernelSize):
            neighbourNonFlashVector = imgNonFlash[nY, nX]
            neighbourFlashVector = imgFlash[nY, nX]

            distCoef = gaussDist(sqrt((x - nX)**2 + (y - nY)**2))

            for i in range(0, channels):
                coefficient = distCoef * intensityCoefficients[abs(neighbourFlashVector[i] - centerFlashVector[i])]
                coefficients[i] = coefficient 
                normalisationFactors[i] += coefficient
                newColours[i] += coefficient * neighbourNonFlashVector[i]


        for i in range(0, channels):
            newColours[i] = newColours[i] / normalisationFactors[i]
        
        return newColours
        
    result = createBlank(width, height)

    for y in range(0, height):
        print(y, height)
        for x in range(0, width):
            result[y, x] = process_pixel(x, y)

    return result

    # ------------------ / Runner
if __name__ == "__main__":
    imgFlash = cv2.imread(INPUT_FLASH, cv2.IMREAD_UNCHANGED)
    imgNoFlash = cv2.imread(INPUT_NON_FLASH, cv2.IMREAD_UNCHANGED)

    if imgFlash is not None:
        sigmaIntensity = 25
        sigmaDistance = 2

        filtered = joint_bilateral_filter(
            imgNoFlash, imgFlash, sigmaDistance, sigmaIntensity, kernelSize=20)

        cv2.imwrite('result.jpg', filtered)
    else:
        print("Couldn't find image " + INPUT_FLASH)
