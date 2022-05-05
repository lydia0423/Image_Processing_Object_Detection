import cv2
import numpy as np

## 0:contour,  1:indexing
def Contour_index_color (img,choice):
    global Object_Largest, Object_Smallest, Largest_ID, Smallest_ID, Total_Object
    numbering = 0
    ID_num = 0

    for c in Contour_Dilated:
        area = cv2.contourArea(c) #Area of a shape.
        if (area <= 500): #if area <=500, not counted.
            continue

        ##compare areas
        ID_num += 1
        if (area > Object_Largest):
            Object_Largest = area
            Largest_ID = ID_num
        if (area < Object_Smallest):
            Object_Smallest = area
            Smallest_ID = ID_num
        perimeter = cv2.arcLength(c,closed=True)#perimeter
        corner_list = cv2.approxPolyDP(c,0.0001 * perimeter,True) #approximate the shape of each object
        coordinates = corner_list.ravel()
        if (choice == 0):
            cv2.drawContours(img,[corner_list],0,color=(0,0,255),thickness=2) #draw contour lines on original pic.

        if (choice == 1):
            ###To index each object
            i = 0
            for j in coordinates:
                if(i % 2 == 0):
                    x = coordinates[i]
                    y = coordinates[i + 1]
                    string = str(x) + " " + str(y)
                    if(i == 0):
                        # text on topmost co-ordinate.
                        numbering += 1
                        cv2.putText(img, "No.{}".format(numbering), (x, y-6),font, 0.5, (255, 0, 0))
                i = i + 1
            Total_Object = numbering
            ###end of indexing

        ##Totally colouring those that fit the condition
        if (len(corner_list) > 60 and choice == 2):
            cv2.fillPoly(img, [corner_list], color=(255, 70, 144))
    return img

##draws circle on img
def circle_find(img):
    global Circles_num
    Circles_num = 0
    gray_blurred = cv2.blur(Dilate_Thresh, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred,cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,param2 = 30, minRadius = 10, maxRadius = 140)
    detected_circles = np.uint16(np.around(detected_circles))
    print("detected",detected_circles)
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv2.circle(img, (a, b), r, (0, 255, 0), 2)
        Circles_num += 1
    return img

##Beautify Presentation
def ShowWindow(title, img, x, y):
    cv2.namedWindow(title)        # Create a named window
    cv2.moveWindow(title, x, y)   # Move it to (x,y)
    cv2.imshow(title,img)

Object_Largest = 0
Object_Smallest = 1000
Largest_ID = 0
Smallest_ID = 0
Total_Object = 0

font = cv2.FONT_HERSHEY_COMPLEX #Type of font for writing.
orig = cv2.imread("image10.png") #read shapes.png
origGray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY) # converting bg to grey
retVal, Thresh_Gray = cv2.threshold(origGray, 254, 255, cv2.THRESH_BINARY_INV) #Only the background white become black the rest becomes white.
To_Dilate = np.copy(Thresh_Gray)
Dilate_Thresh = cv2.morphologyEx(To_Dilate,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=1) #dilation
Contour_Dilated, hierarchy = cv2.findContours(Dilate_Thresh,method=cv2.RETR_LIST,mode=cv2.CHAIN_APPROX_NONE) #contouring all objects


ShowWindow("1 Original" ,orig, 0, 0)
ShowWindow("2 Original-Gray",origGray, 340, 0)
ShowWindow("3 Threshold-Grayed",Thresh_Gray, 680,0)
ShowWindow("4 Dilated-Threshold-Grayed",Dilate_Thresh, 1020, 0)
ShowWindow("5 Img Indexing",Contour_index_color(orig,1), 0, 280) ## indexing
orig2 = np.copy(orig) # orig2 has indexing
ShowWindow("6 Draw Contour",Contour_index_color(orig,0), 340, 280) ## draw contour lines
ShowWindow("7 Colour corners > 60",Contour_index_color(orig,2), 680, 280) ## colour full area > 60

(Contour_index_color(orig2,2)) ## Colour corners > 60
ShowWindow("8 Find Circle", circle_find(orig2), 1020, 280)

print("Number of Objects: ", Total_Object)
print("Largest Object ID: No."+ str(Largest_ID))
print("Largest Object Area: ", Object_Largest)
print("Smallest Object ID: No."+ str(Smallest_ID))
print("Smallest Object Area: ", Object_Smallest)
print("Number of Circle Objects: ", Circles_num)

cv2.waitKey(0)
cv2.destroyAllWindows()
