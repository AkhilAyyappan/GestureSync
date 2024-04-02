
import cv2
import numpy as np
import HandDetectionModule as htm
import time
import autopy

# Constants
wCam, hCam = 640, 480
frame_reduction = 100
smoothening = 7
zoom_factor = 1.1
click_distance_threshold = 40
is_zooming = False

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.HandDetection(max_hands=1)
wScr, hScr = autopy.screen.size()

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    if not success:
        print("Failed to read frame from the camera")
        break
    img = detector.find_hands(img)
    lmList, bbox,img = detector.find_position(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # 3. Check which fingers are up
    fingers = detector.fingers_up()
    cv2.rectangle(img, (frame_reduction, frame_reduction), (wCam - frame_reduction, hCam - frame_reduction),
                  (255, 0, 255), 2)


    # 4. Only Index Finger: Moving Mode
    if fingers[1] == 1 and fingers[2] == 0:
        # 5. Convert Coordinates
        x3 = np.interp(x1, (frame_reduction, wCam - frame_reduction), (0, wScr))
        y3 = np.interp(y1, (frame_reduction, hCam - frame_reduction), (0, hScr))
        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # 7. Move Mouse
        autopy.mouse.move(wScr - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

    # 8. Both Index and middle fingers are up: Clicking Mode
    if fingers[1] == 1 and fingers[2] == 1:
        # 9. Find distance between fingers
        img, distance,lineinfo = detector.find_distance(img,8, 12)
        print(distance)

        # 10. Click mouse if distance is short
        if distance < click_distance_threshold:
            cv2.circle(img, (lineinfo[4], lineinfo[5]), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)