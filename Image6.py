import cv2
import numpy as np


def run():
    pass


def f(x):
    run()
    return x


a = cv2.imread("Image/Image6.jpg")

scale_percent = 10  # percent of original size
width = int(a.shape[1] * scale_percent / 100)
height = int(a.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
a = cv2.resize(a, dim, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

adap_eq = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
gray = adap_eq.apply(gray)

cv2.namedWindow("output")

cv2.createTrackbar("threshold", "output", 122, 255, f)
cv2.createTrackbar("morph_iter", "output", 6, 10, f)


def run():
    thresh = cv2.getTrackbarPos("threshold", "output")

    morph_iter = cv2.getTrackbarPos("morph_iter", "output")

    r, b = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    b = cv2.morphologyEx(
        b,
        cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=morph_iter
    )

    orig = np.copy(a)

    cont, hier = cv2.findContours(b,
                                  method=cv2.RETR_LIST,
                                  mode=cv2.CHAIN_APPROX_SIMPLE)

    triangular_shapes = 0
    for c in cont:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        corner_list = cv2.approxPolyDP(c, 0.02*perimeter, True)

        if len(corner_list) == 3:
            triangular_shapes += 1

        if len(corner_list) > 2:
            cv2.fillPoly(orig, [corner_list], color=(0, 0, 255))

    N, idx, stat, cent = cv2.connectedComponentsWithStats(
        b,
        connectivity=4,
        ltype=cv2.CV_32S
    )

    valid_objects = 0

    # [area, index]
    largest_object = [0, -1]
    smallest_object = [orig.shape[0]*orig.shape[1], -1]
    for index, ([x, y, w, h, area], [cent_x, cent_y]) in enumerate(zip(stat, cent)):
        if area < gray.shape[0] * gray.shape[1] * 0.6:
            cv2.putText(orig,
                        "#{}".format(index),
                        (int(cent_x),
                         int(cent_y)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 0, 0)
                        )

            if area > largest_object[0]:
                largest_object[0] = area
                largest_object[1] = index

            if area < smallest_object[0]:
                smallest_object[0] = area
                smallest_object[1] = index

            valid_objects = valid_objects + 1
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    print("There are {} items.".format(valid_objects))
    print("There are {} triangular shapes.".format(triangular_shapes))
    print("The largest item is item #{}.".format(largest_object[1]))
    print("The smallest item is item #{}.".format(smallest_object[1]))

    cv2.imshow("thresh", b)
    cv2.imshow("output", orig)


run()

cv2.waitKey(0)
cv2.destroyAllWindows()
