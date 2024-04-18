import cv2
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
from tkinter import filedialog
import threading

from real import root

webcam_width, webcam_height = 640, 480


class HandZoomApp:
    def __init__(self):
        self.detector = HandDetector(detectionCon=0.7)
        self.startDist = None
        self.selected_image_path = None

    def select_image(self):
        # root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename()  # Open file dialog to select an image
        return file_path

    def display_selected_image(self, image_path, scale_factor=1):
        if image_path:
            image = cv2.imread(image_path)
            scaled_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
            cv2.imshow("Selected Image", scaled_image)

    def process_video_stream(self):
        cap = cv2.VideoCapture(1)  # Use 0 for default webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)

        while True:
            success, img = cap.read()
            hands, img = self.detector.findHands(img)

            if len(hands) == 2:
                if self.detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and \
                        self.detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
                    if self.startDist is None:
                        length, _, _ = self.detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                        self.startDist = length

                    length, _, _ = self.detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                    scale_factor = 1 + (length - self.startDist) / self.startDist

                    # Display the selected image with zooming
                    self.display_selected_image(self.selected_image_path, scale_factor)

            else:
                self.startDist = None

            try:
                # Get the selected image path if not already selected
                if self.selected_image_path is None:
                    self.selected_image_path = self.select_image()

            except Exception as e:
                print(f"Error: {e}")

            cv2.imshow("Image", img)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()


def main():
    app = HandZoomApp()
    # Start the video processing loop in a separate thread
    video_thread = threading.Thread(target=app.process_video_stream)
    video_thread.start()

    # Start the Tkinter main loop in the main thread
    root = tk.Tk()
    root.mainloop()
    video_thread.join()  # Wait for the video thread to finish before closing


if __name__ == "__main__":
    main()
