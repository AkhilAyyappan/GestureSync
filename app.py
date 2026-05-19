
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from HandTrackingModule import HandDetection


class HandControlApp:
    def __init__(self, root, cap, webcam_width, webcam_height):
        self.root = root
        self.cap = cap
        self.detector = HandDetection()

        # Create GUI elements
        self.canvas = tk.Canvas(root, width=1200, height=720)
        self.canvas.pack()
        self.start_button = ttk.Button(root, text="Start", command=self.start_detection)
        self.start_button.pack()

        self.webcam_width, self.webcam_height = webcam_width, webcam_height

    def start_detection(self):
        detector = HandDetection()  # Initialize HandDetection object

        while True:
            success, image = self.cap.read()

            if not success:
                print("Error reading frame from webcam")
                break

            # Pass image to the `find_hands` method for processing
            image = detector.find_hands(image)

            # Show FPS on the image
            image = detector.show_fps(image)

            # Find hand landmarks and bounding box
            list_of_lm, bbox, image = detector.find_position(image, 0, True)

            # Detect finger positions
            if len(list_of_lm):
                detector.fingers_up()

            image = detector.mode_select(image, self.webcam_width)
            if detector.mode == 0:
                image = detector.volume_controller(image)
            elif detector.mode == 1:
                image = detector.brightness_controller(image)
            elif detector.mode == 2:
                # cv2.circle(image, (webcam_width // 2, webcam_height // 2), 10, (255, 0, 0))
                image = detector.cursor_move(image, self.webcam_width, self.webcam_height)
                detector.click()
                detector.scroll()
                # detector.click_and_drag()
            elif detector.mode == 3:
                image = detector.hand_keyboard(image)

            # Update the canvas with the processed image
            self.display_image(image)

            # Update the GUI
            self.root.update()

            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def display_image(self, image):
        # Convert image from BGR to RGB for tkinter
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)


def main():
    # Set webcam width and height for desired resolution
    webcam_width, webcam_height = 1250, 720

    try:
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    except Exception as e:
        print("Error opening webcam:", e)
        exit()

    root = tk.Tk()
    root.title("GestureSync")

    app = HandControlApp(root, cap, webcam_width, webcam_height)

    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
