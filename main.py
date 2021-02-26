# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import cv2
import numpy as np
from imageproc import throimg

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
print("The training images provided are: ")
dirlist = os.listdir('Imtrain')
print(dirlist)
path = 'Imtrain/'
imagedb = [None for i in range(len(dirlist))]
orb = (cv2.ORB_create(nfeatures=500000))
des = [None for i in range(len(dirlist))]
keyp = [None for i in range(len(dirlist))]

for i in range(len(dirlist)):
    try:
        imagedb[i] = throimg(cv2.imread(path + dirlist[i], 0))
        keyp[i], des[i] = orb.detectAndCompute(imagedb[i], None)
    except:
        pass

testpath = 'Imtest/'
print("The test images provided are: ")
tdirlist = os.listdir('Imtest')
print(tdirlist)
imagedbt = [None for i in range(len(tdirlist))]
dest = [None for i in range(len(tdirlist))]
keypt = [None for i in range(len(tdirlist))]
orbt = (cv2.ORB_create(nfeatures=500000))
for i in range(len(tdirlist)):
    try:
        imagedbt[i] = throimg(cv2.imread(testpath + tdirlist[i], 0))
        keypt[i], dest[i] = orbt.detectAndCompute(imagedbt[i], None)
    except:
        pass

# It's time for the matchmaking ceremony

if len(tdirlist) == []:
    print("Add test images")
else:
    bf = cv2.BFMatcher()
    matchinatrix = [[None for i in range(len(dirlist))] for j in range(len(tdirlist))]
    for i in range(len(dirlist)):
        for j in range(len(tdirlist)):
            print(i, j)
            matches = bf.knnMatch(des[i], dest[j], k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchinatrix[j][i] = good

# print(np.size(matchinatrix))
for i in range(np.size(matchinatrix, 0)):
    print("hi, i is not here at ", i)
    lent = []
    for k in matchinatrix[i]:
        lent.append(len(k))
    varargma = np.argmax(lent)
    if max(lent) >= 10:
        print("The test image", tdirlist[i], "matches the most with", dirlist[varargma])
        showmostmatched = cv2.drawMatchesKnn(imagedbt[i], keypt[i], imagedb[varargma], keyp[varargma],
                                             matchinatrix[i][np.argmax(lent)], None, flags=2)
        cv2.imshow('Most matched ' + str(i + 1), showmostmatched)
        cv2.waitKey(0)
    else:
        print("Matched features for image", tdirlist[i], "are less than 10")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
