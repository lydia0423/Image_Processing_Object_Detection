import cv2
import numpy as np

#function to create the stack
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

#function to get contour
def getContours(img):
    #defining required variables
    global triangle, square, rectangle, circle, totalObjects, maxID, minID
    triangle = 0
    square = 0
    rectangle = 0
    circle = 0
    objectID = 1
    totalObjects = 0
    maxID = 0
    minID = 0
    maxSize = 0
    minSize = 0
    counter = 0

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)     #get the area of objects
        if area > 500:          #only process objects with area greater than 500
            cv2.drawContours(imageContour, cnt, -1, (255, 255, 0), 3)       #draw the contours of objects
            contour_length = cv2.arcLength(cnt, True)         #to get the length of the contour
            approximate = cv2.approxPolyDP(cnt, 0.02 * contour_length, True)    #to get the position of corner points of objects in the image
            objectCorner = len(approximate)             #get the number of corners of objects
            x, y, width, height = cv2.boundingRect(approximate)     #to get the bounding box of objects

            #determining the shape of object
            if objectCorner == 3:
                objectType = "Triangle"
                triangle = triangle + 1
            elif objectCorner == 4:
                aspRatio = width / float(height)
                if aspRatio > 0.98 and aspRatio < 1.03:
                    objectType = "Square"
                    square = square + 1
                else:
                    objectType = "Rectangle"
                    rectangle = rectangle + 1
            elif objectCorner > 4:
                objectType = "Circle"
                circle = circle + 1
            else:
                objectType = "None"

            cv2.rectangle(imageContour, (x, y), (x + width, y + height), (0, 0, 255), 2)         #draw rectangle showing width and height of the objects
            cv2.putText(imageContour, objectType,
                        (x + (width // 2) - 10, y + (height // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)       #to print the shape of objects on the objects
            cv2.putText(imageContour, str(objectID),
                        (x + (width // 2) - 10, y + (height // 2) - 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)       #to put ID of objects

            if counter == 0:
                maxSize = area
                maxID = objectID
                minSize = area
                minID = objectID

            if area > maxSize:
                maxSize = area
                maxID = objectID

            if area < minSize:
                minSize = area
                minID = objectID

            totalObjects = objectID

            objectID = objectID + 1
            counter = counter = 1


address = "C:\\Users\\micah\\OneDrive - Asia Pacific University\\APU Materials\\Year 2 Semester 2\\Imaging & Special Effects\\Assignment\\images\\9.jpg"
image = cv2.imread(address)     #read the image
imageContour = image.copy()     #copy the image to new variable

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)         #make the image gray
imageBlur = cv2.GaussianBlur(imageGray, (7, 7), 1)          #make the image blurry with Gaussian Blur
imageCanny = cv2.Canny(imageBlur, 50, 50)                   #to create canny image
getContours(imageCanny)

imageBlank = np.zeros_like(image)                           #generate blank image
imageStack = stackImages(0.8, ([image, imageGray, imageBlur],
                             [imageCanny, imageContour, imageBlank]))           #creating the stack

print("The total number of objects in the scene is: ", totalObjects)
print("The number of triangular object in the scene is: ", triangle)
print("The number of square object in the scene is: ", square)
print("The number of rectangular object in the scene is: ", rectangle)
print("The number of circular object in the scene is: ", circle)
print("The largest object in the scene is ID: ", maxID)
print("The smallest object in the scene is ID: ", minID)

cv2.imshow("Process", imageStack)
cv2.imshow("Result", imageContour)

cv2.waitKey(0)