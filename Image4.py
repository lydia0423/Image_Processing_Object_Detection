import cv2
import numpy as np


def run():
    pass


def f(x):
    run()
    return x


a = cv2.imread("Image/Image4.jpg")

scale_percent = 15  # percent of original size
width = int(a.shape[1] * scale_percent / 100)
height = int(a.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
a = cv2.resize(a, dim, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("output")

cv2.createTrackbar("threshold", "output", 180, 255, f)


def run():
    thresh = cv2.getTrackbarPos("threshold", "output")

    r, b = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    b = cv2.morphologyEx(
        b,
        cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=1
    )

    N, idx, stat, cent = cv2.connectedComponentsWithStats(b)

    orig = np.copy(a)
    valid_objects = 0
    largest_object = [0, -1]
    smallest_object = [orig.shape[0] * orig.shape[1], -1]
    for index, ([x, y, w, h, area], [cent_x, cent_y]) in enumerate(zip(stat, cent)):
        if area > 100 and area < (orig.shape[0] * orig.shape[1] * 0.8):
            if (area > largest_object[0]):
                largest_object = [area, index]

            if (area < smallest_object[0]):
                smallest_object = [area, index]

            valid_objects += 1

            cv2.rectangle(orig, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.putText(orig,
                        "#{}".format(index),
                        (int(cent_x),
                         int(cent_y)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 255, 255)
                        )

    cv2.imshow("thresh", b)
    cv2.imshow("output", orig)
    print("Number of valid objects: {}".format(valid_objects))
    print("Largest object: object #{}".format(largest_object[1]))
    print("Smallest object: object #{}".format(smallest_object[1]))


run()
cv2.waitKey(0)
