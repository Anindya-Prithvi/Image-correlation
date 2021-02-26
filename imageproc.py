import cv2 as cv


# import matplotlib.pyplot as plt
# import numpy as np


def throimg(img):
    img = cv.resize(img, (320, 240), interpolation=cv.INTER_LINEAR)
    # th1 = cv.adaptiveThreshold(img, 10, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    th1 = img - 10
    th1 = cv.equalizeHist(th1)
    th1 = cv.bilateralFilter(th1, 11, 75, 75)
    th1 = cv.bitwise_not(th1)
    th1 = cv.Canny(th1, 150, 200)

    # plt.imshow(th1,cmap='gray')
    # plt.show()
    return th1
