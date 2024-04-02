import cv2
import numpy as np
import HandDetectionModule as htm
import time
import autopy

wCam, hCam = 640, 480
zoom_factor = 1.1
pTime = 0
click_distance_threshold = 40
is_zooming = False

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.HandDetection(max_hands=1)
wScr, hScr = autopy.screen.size()


while True:

    success, img = cap.read()
    img = detector.find_hands(img)
    lmList, bbox,img = detector.find_position(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    fingers = detector.fingers_up()

    # Zooming Mode(Index and Middle fingers together)
    if fingers[1] == 1 and fingers[2] == 1:
        # 5. Find distance between fingers
        img, distance, lineinfo = detector.find_distance(img, 8, 12)
        # Adjust zoom based on distance (closer = zoom in, farther = zoom out)
        new_zoom_factor = zoom_factor if distance < click_distance_threshold else 1 / zoom_factor

        # Apply zoom if zoom mode is active and zoom factor has changed
        if is_zooming and new_zoom_factor != zoom_factor:
            # Simulate pinch-to-zoom gesture on screen using autopy
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            start_pos = (wScr - center_x, center_y)
            end_pos = (start_pos[0] * new_zoom_factor, start_pos[1] * new_zoom_factor)
            autopy.mouse.move(start_pos[0], start_pos[1])
            autopy.mouse.drag(end_pos[0], end_pos[1], duration=0.1)  # Adjust duration for smoother zoom animation

            # Update zoom factor
            zoom_factor = new_zoom_factor

        # Set zoom mode flag
        is_zooming = True
        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        # 12. Display
        cv2.imshow("Image", img)
        cv2.waitKey(1)