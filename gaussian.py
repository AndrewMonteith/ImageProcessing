'''
    Filename: gaussian.py
    Description: Applies a guassian filter to a given image
    Note: Alongside this file is an example blurred with the inbuit
          opencv so you can check whether the filter works.
'''
import cv2
from math import sqrt, exp, pi
import numpy as np

# --------- / Constants
IMAGE_NAME = 'goodboy.jpg'

# --------- / Custom Blurring


def create_blank(width, height, rgb_color=(0, 0, 0)):
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


def gaussian_blur(img, kernelSize, sigma):
    k, K = 1 / (sqrt(2 * pi) * sigma), 1 / ((-2) * sigma**2)

    def gauss(x):
        return k * exp((x * x) * K)

    (height, width, _) = img.shape
    result = create_blank(width, height)

    def process_pixel(x, y):
        newR, newG, newB = 0, 0, 0
        normal = 0

        for (nX, nY) in neighbourhood(x, y, width, height, kernelSize):
            (b, g, r) = img[nY, nX]
            coef = gauss(sqrt((x - nX)**2 + (y - nY)**2))

            newR += (r * coef)
            newG += (g * coef)
            newB += (b * coef)
            normal += coef

        return (newB / normal, newG / normal, newR / normal)

    for y in range(0, height):
        print("Doing Row:", y)
        for x in range(0, width):
            result[y, x] = process_pixel(x, y)

    return result

# --------- / Entry Point


if __name__ == "__main__":
    img = cv2.imread(IMAGE_NAME, cv2.IMREAD_COLOR)

    if img is not None:

        blurred = gaussian_blur(img, kernelSize=7, sigma=80)
        cv2.imwrite(IMAGE_NAME + "filtered.jpg", blurred)
