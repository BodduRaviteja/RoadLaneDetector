import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit =[]
    right_fit =[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    #resize = cv2.resize(canny, (600, 400))
    #print(resize.shape)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
           x1, y1, x2, y2 = line.reshape(4) #converting into 1-D array from the obtained 2-D array
           cv2.line(line_image, (x1, y1), (x2, y2), (255,0 , 0), 10)
    return line_image


def roi(image):
    h = image.shape[0]
    # Setting an Array of polygons
    ploygon = np.array([
        [(200, h), (1100, h), (550, 250)]
    ])
    mask = np.zeros_like(image)
    # fillPoly fn fills the several polygons not only our defined polygon
    cv2.fillPoly(mask, ploygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


img = cv2.imread('Lane_test_image.jpg ')
#img1 = cv2.resize(img, (600, 400))
#C = canny(img)
#M = roi(C)
#lines = cv2.HoughLinesP(M, 2, (np.pi) / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#averaged_lines = average_slope_intercept(C, lines)
#line_image = display_lines(img, averaged_lines)
# cv2.imshow('L1', img)
# cv2.imshow('gray', gray)
# cv2.imshow('Blur', blur)
# cv2.imshow('Canny', C)
#Combo_image = cv2.addWeighted(line_image, 0.8, img, 1, 1)
#cv2.imshow('ROI', M )
#cv2.imshow('Lanes', line_image)
#cv2.imshow('Final Image', Combo_image)
#cv2.waitKey(0)
#plt.imshow(C)
#plt.show()
#cv2.destroyAllWindows()


#Video Display
cap = cv2.VideoCapture('A:\\OpenCV-Python selbst\\Road Lane Video.mp4')
if (cap.isOpened() == False):
  print("Error opening video stream or file")
# Read until video is completed
while cap.isOpened() == True:
  rate, frame = cap.read()
  if rate == True:
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Frame',frame)
    # img1 = cv2.resize(img, (600, 400))
    C = canny(frame)
    M = roi(C)
    lines = cv2.HoughLinesP(M, 2, (np.pi) / 180, 100, np.array([]), minLineLength=40, maxLineGap=200)
    averaged_lines = average_slope_intercept(C, lines)
    line_image = display_lines(frame, averaged_lines)
    Combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('Lanes', line_image)
    cv2.imshow('Final Image', Combo_image)
    if cv2.waitKey(1) & 0xFF == ord('t'):
      break
  else:
    break
# When everything done, release the video capture object
cap.release()
cv2.destroyAllWindows()