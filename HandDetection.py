import cv2
import mediapipe as mp
import autopy

wcam, hcam = 640, 480
cap = cv2.VideoCapture(0)  # Use 0 instead of 1 for the default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wcam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hcam)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set the offset for the camera window placement
screen_width, screen_height = autopy.screen.size()
offset_x = screen_width - wcam

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        success, img = cap.read()

        # Convert the image to RGB
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe hand detection
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the image
        cv2.imshow("Image", img)

        # Move the mouse cursor to the right side of the screen
        autopy.mouse.move(offset_x, 0)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()
