import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
cam = cv2.VideoCapture(0)
time.sleep(2)
bg = 0

for i in range(60):
    ret, bg = cam.read()

bg = np.flip(bg, axis = 1)

while (cam.isOpened()):
    ret, img = cam.read()
    if not ret:
        break
    img = np.flip(img, axis = 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 += mask2
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    mask2 = cv2.bitwise_not(mask1)
    res1 = cv2.bitwise_and(img, img, mask = mask2)
    res2 = cv2.bitwise_and(bg, bg, mask = mask1)

    final = cv2.addWeighted(res1, 1, res2, 1, 0)
    output.write(final)
    cv2.imshow('Invisibility', final)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()