import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import numpy as np
import pyautogui

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pTime = 0
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Get the screen size
screen_width, screen_height = pyautogui.size()

def fingers_up(lmList):
    fingers = []

    if lmList[4][0] > lmList[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    for id in range(8, 21, 4):
        if lmList[id][1] < lmList[id - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    img = cv2.resize(img, (640, 480))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])
    mask = cv2.inRange(imgHSV, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    hands, img = detector.findHands(img, draw=True)

    if hands:
        lmList = hands[0]['lmList']
        if len(lmList) >= 21:
            index_finger = lmList[8]
            middle_finger = lmList[12]
            if index_finger and middle_finger:
                index_x, index_y, _ = index_finger
                middle_x, middle_y, _ = middle_finger
                index_movement = (index_x, index_y)
                middle_movement = (middle_x, middle_y)
                print(f"Index finger movement: {index_movement}")
                print(f"Middle finger movement: {middle_movement}")
            
            fingers = fingers_up(lmList)
            print(f"Fingers up: {fingers}")
            for i, finger in enumerate(fingers):
                if finger == 1:
                    print(f"Finger {i+1} is up")

            # Example usage of screen_width and screen_height
            if fingers[1] == 1 and fingers[2] == 0:
                x1, y1 = lmList[8][:2]
                x3 = np.interp(x1, (0, 640), (0, screen_width))
                y3 = np.interp(y1, (0, 480), (0, screen_height))
                pyautogui.moveTo(screen_width - x3, y3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
