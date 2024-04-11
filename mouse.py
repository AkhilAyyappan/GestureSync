import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

# Constants
wCam, hCam = 640, 480
frame_reduction = 100
smoothening = 7
click_distance_threshold = 40
scroll_distance_threshold = 30
action_cooldown = 0.3

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
mode = "move"  # Initial mode is moving
action_time = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.HandDetection(max_hands=1, detection_confidence=0.5, tracking_confidence=0.5)  # Adjust confidence levels as needed
wScr, hScr = autopy.screen.size()

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.find_hands(img)

    lmList, bbox, img = detector.find_position(img)

    fingers = None
    if len(lmList):
        # 2. Check which fingers are up
        fingers = detector.fingers_up()
        cv2.rectangle(img, (frame_reduction, frame_reduction), (wCam - frame_reduction, hCam - frame_reduction),
                      (255, 0, 255), 2)

        # 3. Index Finger: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            mode = "move"
            x1, y1 = lmList[8][1:]
            x = np.interp(x1, (frame_reduction, wCam - frame_reduction), (0, wScr))
            y = np.interp(y1, (frame_reduction, hCam - frame_reduction), (0, hScr))
            clocX = plocX + (x - plocX) / smoothening
            clocY = plocY + (y - plocY) / smoothening
            autopy.mouse.move(clocX, clocY)
            plocX, plocY = clocX, clocY

        # 4. Index and middle fingers are up: Clicking Mode
        elif fingers[1] == 1 and fingers[2] == 1:
            mode = "click"
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Click or right-click if distance is short
            if distance < click_distance_threshold:
                autopy.mouse.click()
                action_time = time.time()  # Record the time of the action
            elif distance > click_distance_threshold * 2:
                autopy.mouse.click(autopy.mouse.Button.RIGHT)
                action_time = time.time()  # Record the time of the action

        # 5. Three Fingers are up: Scroll Mode
        elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1:
            mode = "scroll"
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            distance_y = y2 - y1

            # Scroll up or down based on finger movement
            if distance_y > scroll_distance_threshold:
                autopy.mouse.scroll(3)  # Scroll up
                action_time = time.time()  # Record the time of the action
            elif distance_y < -scroll_distance_threshold:
                autopy.mouse.scroll(-3)  # Scroll down
                action_time = time.time()  # Record the time of the action

        # 6. Thumb and Pinky fingers are up: Right-click Mode
        elif fingers[0] == 1 and fingers[4] == 1:
            mode = "right_click"
            # Perform right-click action
            autopy.mouse.click(autopy.mouse.Button.RIGHT)
            action_time = time.time()  # Record the time of the action

    # 7. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # 8. Check for cooldown time after an action to avoid collision
    if time.time() - action_time < action_cooldown:  # Adjust cooldown time as needed
        mode = "none"

    # 9. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
