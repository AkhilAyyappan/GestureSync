import cv2
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
from tkinter import filedialog
import threading

webcam_width, webcam_height = 640, 480

def select_image():
    # root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open file dialog to select an image
    return file_path

def display_selected_image(selected_image_path, scale_factor=1):
    if selected_image_path:
        image = cv2.imread(selected_image_path)
        scaled_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        cv2.imshow("Selected Image", scaled_image)

def process_video_stream():
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)

    detector = HandDetector(detectionCon=0.7)
    startDist = None
    selected_image_path = None

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        if len(hands) == 2:
            if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and \
                    detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
                if startDist is None:
                    length, _, _ = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                    startDist = length

                length, _, _ = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                scale_factor = 1 + (length - startDist) / startDist

                # Display the selected image with zooming
                display_selected_image(selected_image_path, scale_factor)

        else:
            startDist = None

        try:
            # Get the selected image path if not already selected
            if selected_image_path is None:
                selected_image_path = select_image()

        except Exception as e:
            print(f"Error: {e}")

        cv2.imshow("Image", img)
        cv2.waitKey(1)

# Start the video processing loop in a separate thread
video_thread = threading.Thread(target=process_video_stream)
video_thread.start()

# Start the Tkinter main loop in the main thread
root = tk.Tk()
root.mainloop()