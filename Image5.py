import enum
from logging.config import valid_ident
import cv2
import numpy as np


def run():
    pass


def f(x):
    run()
    return x


a = cv2.imread("Image/Image5")

scale_percent = 25  # percent of original size
width = int(a.shape[1] * scale_percent / 100)
height = int(a.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
a = cv2.resize(a, dim, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("output")
cv2.namedWindow("thresh")
cv2.createTrackbar("thresh", "output", 187, 255, f)
cv2.createTrackbar("k", "output", 2, 11, f)
cv2.createTrackbar("sig", "output", 2, 11, f)
cv2.createTrackbar("morph_iter", "output", 0, 10, f)
cv2.createTrackbar("clip_limit", "output", 24, 255, f)
cv2.createTrackbar("tile_grid_size", "output", 4, 255, f)


def run():
    thresh = cv2.getTrackbarPos("thresh", "output")
    k = cv2.getTrackbarPos("k", "output")
    sig = cv2.getTrackbarPos("sig", "output")
    morph_iter = cv2.getTrackbarPos("morph_iter", "output")
    clip_limit = cv2.getTrackbarPos("clip_limit", "output")
    tile_grid_size = cv2.getTrackbarPos("tile_grid_size", "output")

    adap_eq = cv2.createCLAHE(clipLimit=clip_limit,
                              tileGridSize=(tile_grid_size, tile_grid_size))
    res = adap_eq.apply(gray)

    if(k % 2 == 0):
        k = k+1

    res = cv2.GaussianBlur(res, (k, k), sig)

    r, b = cv2.threshold(res, thresh, 255, cv2.THRESH_BINARY_INV)

    b = cv2.morphologyEx(
        b,
        cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=morph_iter
    )

    N, idx, stat, cent = cv2.connectedComponentsWithStats(b)

    orig = np.copy(a)

    valid_objects = 0
    largest_object = [0, -1]
    smallest_object = [orig.shape[0] * orig.shape[1], -1]
    for index, ([x, y, w, h, area], [cent_x, cent_y]) in enumerate(zip(stat, cent)):
        if area < orig.shape[0] * orig.shape[1] * 0.3:
            if (area > largest_object[0]):
                largest_object = [area, index]

            if (area < smallest_object[0]):
                smallest_object = [area, index]

            valid_objects += 1

            cv2.putText(orig,
                        "#{}".format(index),
                        (int(cent_x),
                         int(cent_y)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 0, 255)
                        )

    cv2.imshow("output", orig)
    cv2.imshow("thresh", b)

    print("Number of valid objects: {}".format(valid_objects))
    print("Largest object: object #{}".format(largest_object[1]))
    print("Smallest object: object #{}".format(smallest_object[1]))


cv2.moveWindow("output", 0, 0)
cv2.moveWindow("thresh", 0, 0)

run()
cv2.waitKey(0)
cv2.destroyAllWindows()
