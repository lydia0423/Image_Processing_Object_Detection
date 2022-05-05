import cv2
import numpy as np

def Contour_index_color (img,choice):
    numbering = 0
    ID_num = 0
    global Object_Largest, Object_Smallest, Largest_ID, Smallest_ID, Total_Object, Total_Square, Total_Triangle
    for c in Contour_Dilated:
        area = cv2.contourArea(c)  # Area of a shape.
        if (area <= 3000 or area >= 50000):
            continue

        ##compare areas
        ID_num += 1
        if (area > Object_Largest):
            Object_Largest = area
            Largest_ID = ID_num
        if (area < Object_Smallest):
            Object_Smallest = area
            Smallest_ID = ID_num

        perimeter = cv2.arcLength(c, closed=True)  # perimeter
        corner_list = cv2.approxPolyDP(c, 0.088 * perimeter, True)  # approximate the shape of each object
        coordinates = corner_list.ravel() # coordinates of each corner

        if (choice == 0):
            cv2.drawContours(img, [corner_list], 0, color=(255,0,0), thickness=2)  # draw contour lines on original pic.

        if (choice == 1):
            if (len(corner_list) == 3): ## if triangle
                cv2.fillPoly(img, [corner_list], color=(0, 255, 0)) ##colour full if fits
                Total_Triangle += 1

        if (choice == 2):
            if (len(corner_list) == 4): ## if square
                cv2.fillPoly(img, [corner_list], color=(255, 255, 255)) ##colour full if fits
                Total_Square += 1

        if (choice ==3):
            ###To index each object
            i = 0
            for j in coordinates:
                if (i % 2 == 0):
                    x = coordinates[i]
                    y = coordinates[i + 1]
                    string = str(x) + " " + str(y)
                    if (i == 0):
                        # text on topmost co-ordinate.
                        numbering += 1
                        if (numbering == 1 or numbering == 7): ## No1 and No7 too astray from their actual object.
                            cv2.putText(img, "No.{}".format(numbering), (x + 80, y - 20), font, 0.5, (255, 0, 0))
                        elif (numbering == 2): ##No.2 is unlike the others.
                            cv2.putText(img, "No.{}".format(numbering), (x - 45, y + 30), font, 0.5, (255, 0, 0))
                        else:
                            cv2.putText(img, "No.{}".format(numbering), (x - 70, y +30), font, 0.5, (255, 0, 0))
                i = i + 1
            Total_Object = numbering
            ###end of indexing
    return img

##Beautify Presentation
def ShowWindow(title, img, x, y):
    cv2.namedWindow(title)        # Create a named window
    cv2.moveWindow(title, x, y)   # Move it to (x,y)
    cv2.imshow(title,img)


Object_Largest = 0
Object_Smallest = 10000
Largest_ID = 0
Smallest_ID = 0
Total_Object = 0
Total_Square = 0
Total_Triangle = 0

font = cv2.FONT_HERSHEY_COMPLEX #Type of font for writing.
orig = cv2.imread("image11.png") #read shapes.png
orig2 = np.copy(orig)
origGray = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY) # converting bg to grey
retVal, Thresh_Gray = cv2.threshold(origGray, 125, 255, cv2.THRESH_BINARY_INV) #Only the background white become black the rest becomes white.
To_Dilate = np.copy(Thresh_Gray)
Dilation = cv2.morphologyEx(To_Dilate,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=2)
Contour_Dilated, hierarchy = cv2.findContours(Dilation,method=cv2.RETR_LIST,mode=cv2.CHAIN_APPROX_NONE) #contouring all objects


ShowWindow("1 Original",orig, 0, 0)
ShowWindow("2 Grayed",origGray, 340, 0)
ShowWindow("3 Thresholded",Thresh_Gray, 680, 0)
ShowWindow("4 Dilated",Dilation, 1020, 0)
ShowWindow("5 Contour",Contour_index_color(orig2,0), 0, 385)
ShowWindow("6 Indexing",Contour_index_color(orig2,3), 340, 385)
orig3 = np.copy(orig2)
ShowWindow("7 Triangles",Contour_index_color(orig2,1), 680, 385) ## colour full triangle
ShowWindow("8 Squares",Contour_index_color(orig3,2), 1020, 385) ## colour full squares

print("Number of Objects: ", Total_Object)
print("Largest Object ID: No."+ str(Largest_ID))
print("Largest Object Area: ", Object_Largest)
print("Smallest Object ID: No."+ str(Smallest_ID))
print("Smallest Object Area: ", Object_Smallest)
print("Number of Triangle Objects: ", Total_Triangle)
print("Number of Square Objects: ", Total_Square)

cv2.waitKey(0)
cv2.destroyAllWindows()
