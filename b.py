import math
import time
import cv2
import mediapipe as mp
import numpy as np
# for volume control
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# for brightness control
import screen_brightness_control as sbc

import autopy
import pyautogui

import keyboard


class Button:
    def __init__(self, pos, text, size=(85 * 2, 85)):
        self.pos = pos
        self.size = size
        self.text = text


class HandDetection:
    def __init__(self, mode=False, max_hands=1, model_complexity=1,
                 detection_confidence=0.7, tracking_confidence=0.5):
        self.previous_time = 0
        self.current_time = None
        self.list_of_lm = None
        self.results = None
        self.tipIds = [4, 8, 12, 16, 20]
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,
                                         self.model_complexity, self.detection_confidence,
                                         self.tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils
        self.bbox = []
        self.fingers = None
        self.plocX, self.plocY = 0, 0

        self.caps = 1
        self.keys = [
            [
                ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+"],
                ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "<--"],
                ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":", "\"", "ENT"],
                ["Z", "X", "C", "V", "B", "N", "M", "<", ">", "?", "CAP"],
                ["SPC"]
            ],
            [
                ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="],
                ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "<--"],
                ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "'", "ENT"],
                ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/", "CAP"],
                ["SPC"]
            ]
        ]

    def find_hands(self, image, draw=True):
        # Convert the image to RGB format
        image_in_rgb_format = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe hands detection
        self.results = self.hands.process(image_in_rgb_format)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks,
                                                   self.mp_hands.HAND_CONNECTIONS)

        return image

    def find_position(self, image, hand_no=0, draw=True):
        list_x = []
        list_y = []

        self.list_of_lm = []
        if self.results.multi_hand_landmarks:
            # select a hand
            my_hand = self.results.multi_hand_landmarks[hand_no]
            # store hand landmark position in to a list
            for ID, LM in enumerate(my_hand.landmark):
                h, w, c = image.shape
                cx, cy = int(LM.x * w), int(LM.y * h)
                list_x.append(cx)
                list_y.append(cy)
                self.list_of_lm.append([ID, cx, cy])

                # highlight the hand
                if draw:
                    cv2.circle(image, (cx, cy), 5, (255, 0, 255), 1)
            x_min, x_max = min(list_x), max(list_x)
            y_min, y_max = min(list_y), max(list_y)
            self.bbox = [x_min, y_min, x_max, y_max]
            if draw:
                cv2.rectangle(image, (self.bbox[0] - 10, self.bbox[1] - 10), (self.bbox[2] + 10, self.bbox[3] + 10),
                              (0, 255, 0), 2)
        return self.list_of_lm, self.bbox, image

    def fingers_up(self):
        self.fingers = []
        # Thumb
        if self.list_of_lm[self.tipIds[0]][1] > self.list_of_lm[self.tipIds[0] - 1][1]:
            self.fingers.append(1)
        else:
            self.fingers.append(0)
        # 4 Fingers
        for ID in range(1, 5):
            if self.list_of_lm[self.tipIds[ID]][2] < self.list_of_lm[self.tipIds[ID] - 2][2]:
                self.fingers.append(1)
            else:
                self.fingers.append(0)
        return self.fingers

    def show_fps(self, image):
        # Calculate FPS (frames per second)
        self.current_time = time.time()
        fps = 1 / (self.current_time - self.previous_time)
        self.previous_time = self.current_time

        # Display FPS on image (properly indented)
        cv2.putText(image, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        return image

    def find_distance(self, image, point_1, point_2, draw=True):

        x1, y1 = self.list_of_lm[point_1][1], self.list_of_lm[point_1][2]
        x2, y2 = self.list_of_lm[point_2][1], self.list_of_lm[point_2][2]

        # find center of line and colour it
        cx, cy = (x2 + x1) // 2, (y2 + y1) // 2

        if draw:
            cv2.circle(image, (x1, y1), 5, (255, 0, 255), 5, cv2.FILLED)  # colour thumb fingertip
            cv2.circle(image, (x2, y2), 5, (255, 0, 255), 5, cv2.FILLED)  # colour index fingertip
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 3)  # draw line btw the points
            cv2.circle(image, (cx, cy), 5, (255, 0, 255), 5, cv2.FILLED)

        # distance btw the two points
        distance = math.hypot(x1 - x2, y1 - y2)

        return image, distance, [x1, y1, x2, y2, cx, cy]

    def volume_controller(self, image, draw=True):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = interface.QueryInterface(IAudioEndpointVolume)

        if len(self.list_of_lm):

            # do following if finger 2,3 is down and 4 is up
            if self.fingers[4] and not self.fingers[2] and not self.fingers[3]:
                # filter based on size
                area = ((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])) // 100

                if 150 < area < 800:

                    # draw line btw thump and index, find distance btw them
                    image, distance, line_info = self.find_distance(image, 4, 8, draw=draw)

                    # covert volume
                    vol_bar = np.interp(distance, [50, 180], [400, 150])
                    vol_per = np.interp(distance, [50, 180], [0, 100])

                    # reduce resolution to make smoother
                    smoothness = 10
                    vol_per = round(int(vol_per) / smoothness) * smoothness

                    # set volume
                    volume.SetMasterVolumeLevelScalar(vol_per / 100, None)
                    print("volume control")

                    # drawing
                    if draw:
                        if distance < 50:  # colour center point green/red when distance is min/max
                            cv2.circle(image, (line_info[4], line_info[5]), 5, (0, 255, 0), 5, cv2.FILLED)
                        elif distance >= 180:
                            cv2.circle(image, (line_info[4], line_info[5]), 5, (0, 0, 255), 5, cv2.FILLED)

                        cv2.putText(image, f'{int(vol_per)} %',
                                    (50, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # volume percentage on screen

                        cv2.rectangle(image, (50, 150), (85, 400), (255, 0, 0), 3)  # volume meter on screen
                        cv2.rectangle(image, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)

                        current_volume = int(volume.GetMasterVolumeLevelScalar() * 100)
                        cv2.putText(image, f'Volume: {current_volume}', (400, 50), cv2.FONT_HERSHEY_PLAIN,
                                    2, (255, 0, 0), 2)

        return image

    def brightness_controller(self, image, draw=True):

        if len(self.list_of_lm):

            # do following if finger 3,4 is down and 2 is up
            if not self.fingers[1] and not self.fingers[2] and not self.fingers[3]:
                # filter based on size
                area = ((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])) // 100

                if 150 < area < 800:

                    # draw line btw thump and index, find distance btw them
                    image, distance, line_info = self.find_distance(image, 4, 20, draw=draw)

                    # covert brightness
                    bri_bar = np.interp(distance, [50, 180], [400, 150])
                    bri_per = np.interp(distance, [50, 180], [0, 100])

                    # reduce resolution to make smoother
                    smoothness = 10
                    bri_per = round(int(bri_per) / smoothness) * smoothness

                    # set brightness
                    sbc.set_brightness(bri_per)
                    print("brightness control")
                    # drawing
                    if draw:
                        if distance < 50:  # colour center point green/red when distance is min/max
                            cv2.circle(image, (line_info[4], line_info[5]), 5, (0, 255, 0), 5, cv2.FILLED)
                        elif distance >= 180:
                            cv2.circle(image, (line_info[4], line_info[5]), 5, (0, 0, 255), 5, cv2.FILLED)

                        cv2.putText(image, f'{int(bri_per)} %',
                                    (50, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255),
                                    2)  # volume percentage on screen

                        cv2.rectangle(image, (50, 150), (85, 400), (255, 0, 255), 3)  # volume meter on screen
                        cv2.rectangle(image, (50, int(bri_bar)), (85, 400), (255, 0, 255), cv2.FILLED)

                        current_brightness = sbc.get_brightness()
                        cv2.putText(image, f'Brightness: {current_brightness}', (400, 50), cv2.FONT_HERSHEY_PLAIN,
                                    2, (255, 0, 255), 2)

        return image

    # def cursor_move(self, webcam_width, webcam_height):
    #     wScr, hScr = autopy.screen.size()
    #
    #     frame_reduction = 100
    #     smoothening = 7
    #
    #     if len(self.list_of_lm):
    #         if self.fingers[1] == 1 and self.fingers[2] == 1 and self.fingers[3] == 0 and self.fingers[4] == 0 and \
    #                 self.fingers[0] == 0:
    #             mode = "move"
    #             x1, y1 = self.list_of_lm[9][1:]
    #             x = np.interp(x1, (frame_reduction, webcam_width - frame_reduction), (0, wScr))
    #             y = np.interp(y1, (frame_reduction, webcam_height - frame_reduction), (0, hScr))
    #             clocX = self.plocX + (x - self.plocX) / smoothening
    #             clocY = self.plocY + (y - self.plocY) / smoothening
    #             autopy.mouse.move(clocX, clocY)
    #             self.plocX, self.plocY = clocX, clocY

    def cursor_move(self, webcam_width, webcam_height, draw=True):
        centre_x = webcam_width // 2
        centre_y = webcam_height // 2
        # if draw:
        #     cv2.circle(image, (centre_x, centre_y), 3, (255, 0, 0), 2, cv2.FILLED)
        if len(self.list_of_lm):
            if self.fingers[1] == 1 and self.fingers[2] == 1 and self.fingers[3] == 0 and self.fingers[4] == 0 and \
                    self.fingers[0] == 0:
                current_x, current_y = self.list_of_lm[self.tipIds[1]][1], self.list_of_lm[self.tipIds[1]][2]
                dx = 100 if current_x > centre_x else -100
                dy = 50 if current_y > centre_y else -50
                pyautogui.FAILSAFE = False
                pyautogui.moveRel(dx, dy, 0)

    def click(self):
        if len(self.list_of_lm):
            if self.fingers[0] == 0 and self.fingers[1] == 0 and self.fingers[2] == 1 and self.fingers[3] == 0 and \
                    self.fingers[4] == 0:

                # autopy.mouse.click()
                pyautogui.click(x=None, y=None, button='left', clicks=1, interval=0.3)
                print("left click")
                # self.wait(.3)

            elif self.fingers[0] == 0 and self.fingers[1] == 1 and self.fingers[2] == 0 and self.fingers[3] == 0 and \
                    self.fingers[4] == 0:
                # autopy.mouse.click(autopy.mouse.Button.RIGHT)
                pyautogui.click(x=None, y=None, button='right', clicks=1, interval=0.3)
                print("right click")
                # self.wait(.3)

            elif self.fingers[0] == 0 and self.fingers[1] == 1 and self.fingers[2] == 1 and self.fingers[3] == 0 and \
                    self.fingers[4] == 1:
                pyautogui.doubleClick(x=None, y=None, interval=0.5)
                print("double click")

    def scroll(self):
        if len(self.list_of_lm):
            if not self.fingers[0] and self.fingers[1] and self.fingers[2] and self.fingers[3]:
                if self.fingers[4]:
                    pyautogui.scroll(120)
                    print("scroll up")
                else:
                    pyautogui.scroll(-120)
                    print("scroll down")

    # def click_and_drag(self):
    #     if len(self.list_of_lm):
    #         x, distance, y = self.find_distance(None, 4, 8, False)
    #         if self.fingers[2] and self.fingers[3] and self.fingers[4]:
    #             if distance < 30:
    #
    #                 print("drag")

    # def wait(self, seconds):
    #     current_time = time.time()
    #     future_time = current_time + seconds
    #     while time.time() < future_time:
    #         pass  # Empty statement (does nothing)

    # keyboard
    def cornerRect(self, img, bbox, l=30, t=5, rt=1,
                   colorR=(255, 0, 255), colorC=(0, 255, 0)):
        """
        :param img: Image to draw on.
        :param bbox: Bounding box [x, y, w, h]
        :param l: length of the corner line
        :param t: thickness of the corner line
        :param rt: thickness of the rectangle
        :param colorR: Color of the Rectangle
        :param colorC: Color of the Corners
        :return:
        """
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        if rt != 0:
            cv2.rectangle(img, bbox, colorR, rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), colorC, t)
        cv2.line(img, (x, y), (x, y + l), colorC, t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), colorC, t)
        cv2.line(img, (x1, y), (x1, y + l), colorC, t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), colorC, t)
        cv2.line(img, (x, y1), (x, y1 - l), colorC, t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), colorC, t)
        cv2.line(img, (x1, y1), (x1, y1 - l), colorC, t)

        return img

    def drawAll(self, img, buttonList):
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            self.cornerRect(img, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                            20, rt=0)
            cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(img, button.text, (x + 20, y + 65),
                        cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
        return img

    def assign(self):
        button_list = []
        for i in range(len(self.keys[self.caps])):
            for j, key in enumerate(self.keys[self.caps][i]):
                if len(key) == 1:
                    button_list.append(Button([100 * j + 50, 100 * i + 50], key, (85, 85)))
                else:
                    button_list.append(Button([100 * j + 50, 100 * i + 50], key))
        return button_list

    def hand_keyboard(self, image):
        button_list = self.assign()
        image = self.drawAll(image, button_list)
        if self.list_of_lm:
            if not self.caps:
                keyboard.press("shift")
            # print(button_list[0].text, button_list[0].size, button_list[0].pos)
            # print(lmList[8][1], lmList[8][2])
            for button in button_list:
                x, y = button.pos
                w, h = button.size

                if x < self.list_of_lm[8][1] < x + w and y < self.list_of_lm[8][2] < y + h:
                    cv2.rectangle(image, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                    cv2.putText(image, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                    # print(l)

                    _, l, _ = self.find_distance(image, 8, 12, draw=False)
                    # when clicked
                    if l < 50:
                        if len(button.text) == 1:
                            keyboard.send(button.text)
                        elif button.text == "SPC":
                            keyboard.press("space")
                        elif button.text == "<--":
                            keyboard.press("backspace")
                        elif button.text == "ENT":
                            keyboard.press("enter")
                        elif button.text == "CAP":
                            self.caps = (self.caps + 1) % 2
                            # if keyboard.is_pressed("caps_lock"):
                            #     keyboard.press("caps_lock")
                            # else:
                            #     keyboard.release("caps_lock")
                            time.sleep(0.2)
                        print(button.text)
                        cv2.rectangle(image, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        # finalText += button.text
                        time.sleep(0.3)
            if not self.caps:
                keyboard.release("shift")
        return image


def main():
    # Set webcam width and height for desired resolution
    webcam_width, webcam_height = autopy.screen.size()

    try:
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    except Exception as e:
        print("Error opening webcam:", e)
        exit()

    detector = HandDetection()

    while True:
        success, image = cap.read()

        if not success:
            print("Error reading frame from webcam")
            break

        # Pass image to the `find_hands` method for processing
        image = detector.find_hands(image)

        image = detector.show_fps(image)

        # print hand landmark to terminal
        list_of_lm, bbox, image = detector.find_position(image, 0, True)
        if len(list_of_lm):
            detector.fingers_up()

        # image = detector.volume_controller(image)
        # image = detector.brightness_controller(image)

        # detector.cursor_move(webcam_width, webcam_height)
        # detector.click()
        # detector.scroll()
        # detector.click_and_drag()

        image = detector.hand_keyboard(image)

        # Display the image
        cv2.imshow("Image", image)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
