import cv2
import numpy as np

allArea = []
cntList = []
index = 0
largestId = 0
smallestId = 0
noOfTriangle = 0

image = cv2.imread("Image/Image3.png")
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.GaussianBlur(img_gray, (7,7), 4)

ret, thresh = cv2.threshold(img_gray, 215, 255, cv2.THRESH_BINARY_INV)
d = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
cv2.morphologyEx(thresh, cv2.MORPH_DILATE, d)

(cnt, hierarchy) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, cnt, -1, (0,255,0), 2)

for i in cnt:
    area = cv2.contourArea(i)
    if(area != 0):
        allArea.append(area)
        cntList.append(i)

sortedContours = sorted(cntList, key=cv2.contourArea, reverse=True)
largestItem = sortedContours[0]
smallestItem = sortedContours[-1]

for i in cntList:
    index += 1
    approx = cv2.approxPolyDP(i, 0.009 * cv2.arcLength(i, True), True)
    n = approx.ravel()
    k = 0

    if len(approx) == 3:
        noOfTriangle+=1

    for j in n:
        if (k % 2 == 0):
            x = n[k]
            y = n[k + 1]

            # String containing the co-ordinates.
            string = str(x) + " " + str(y)

            if (k == 0):
                # text on topmost co-ordinate.
                cv2.putText(image, "#{}".format(index), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
                if(np.array_equal(i, largestItem)):
                    largestId = index
                elif(np.array_equal(i, smallestItem)):
                    smallestId = index

cv2.drawContours(image=image, contours=largestItem, contourIdx=-1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)
cv2.drawContours(image=image, contours=smallestItem, contourIdx=-1, color=(255,0,255), thickness=2, lineType=cv2.LINE_AA)

print("The number of objects in the scene is : ", index)
print("The number of objects in triangle shape are : ", noOfTriangle)
print("The smallest object id is : #", smallestId)
print("The largest object id is : #", largestId)

cv2.imshow("Binary image", thresh)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()




