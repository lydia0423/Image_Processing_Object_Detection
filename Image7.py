import cv2
import numpy as np


def run():
    pass


def f(x):
    run()
    return x


a = cv2.imread("Image/Image7.jpg")

scale_percent = 10  # percent of original size
width = int(a.shape[1] * scale_percent / 100)
height = int(a.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
a = cv2.resize(a, dim, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("output")
cv2.namedWindow("thresh")
cv2.createTrackbar("thresh", "output", 190, 255, f)
cv2.createTrackbar("k", "output", 5, 11, f)
cv2.createTrackbar("sig", "output", 2, 11, f)
cv2.createTrackbar("morph_iter", "output", 2, 10, f)


def run():
    thresh = cv2.getTrackbarPos("thresh", "output")
    k = cv2.getTrackbarPos("k", "output")
    sig = cv2.getTrackbarPos("sig", "output")
    morph_iter = cv2.getTrackbarPos("morph_iter", "output")

    b = cv2.equalizeHist(gray)

    if(k % 2 == 0):
        k = k+1

    res = cv2.GaussianBlur(b, (k, k), sig)

    r, b = cv2.threshold(res, thresh, 255, cv2.THRESH_BINARY)

    b = cv2.morphologyEx(
        b,
        cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=morph_iter
    )

    N, idx, stat, cent = cv2.connectedComponentsWithStats(b)

    orig = np.copy(a)

    for [x, y, w, h, area] in stat:
        cv2.rectangle(orig, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)

    cv2.imshow("output", orig)
    cv2.imshow("thresh", b)


cv2.moveWindow("output", 0, 0)
cv2.moveWindow("thresh", 0, 0)

run()
cv2.waitKey(0)
cv2.destroyAllWindows()
