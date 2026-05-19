
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
                ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":", "\""],
                ["Z", "X", "C", "V", "B", "N", "M", "<", ">", "?", "CAP"],
                ["SPC"],
                ["ENT"]
            ],
            [
                ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="],
                ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "<--"],
                ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "'"],
                ["z", "x", "c", "v", "b", "n", "m", ",", ".", "/", "CAP"],
                ["SPC"],
                ["ENT"]
            ]
        ]

        self.mode = 0

        self.previous_position_x = 1200//2
        self.previous_position_y = 720//2

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

                if 150 < area < 1000:

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
                        cv2.putText(image, f'Volume: {current_volume}', (1000, 50), cv2.FONT_HERSHEY_PLAIN,
                                    2, (255, 0, 0), 2)

        return image

    def brightness_controller(self, image, draw=True):

        if len(self.list_of_lm):

            # do following if finger 3,4 is down and 2 is up
            if self.fingers[4] and not self.fingers[2] and not self.fingers[3]:
                # filter based on size
                area = ((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])) // 100

                if 150 < area < 1000:

                    # draw line btw thump and index, find distance btw them
                    image, distance, line_info = self.find_distance(image, 4, 8, draw=draw)

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
                        cv2.putText(image, f'Brightness: {current_brightness}', (950, 50), cv2.FONT_HERSHEY_PLAIN,
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

    def cursor_move(self, image, webcam_width, webcam_height):
        # centre_x = webcam_width // 2
        # centre_y = webcam_height // 2

        # rect_width = 200
        # rect_height = 100
        # center_x = int(webcam_width / 2)
        # center_y = int(webcam_height / 2)
        # # Calculate top-left corner coordinates
        # top_left_x = center_x - int(rect_width / 2)
        # top_left_y = center_y - int(rect_height / 2)
        # bottom_right_x = top_left_x + rect_width
        # bottom_right_y = top_left_y + rect_height
        # # Draw rectangle using cv2.rectangle
        # cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y),
        #               (0, 255, 0), 2)  # Green rectangle with thickness 2

        if len(self.list_of_lm):
            if self.fingers[1] == 1 and self.fingers[2] == 1 and self.fingers[3] == 0 and self.fingers[4] == 0 and \
                    self.fingers[0] == 0:
                current_x, current_y = self.list_of_lm[self.tipIds[1]][1], self.list_of_lm[self.tipIds[1]][2]

                # if top_left_x < current_x < bottom_right_x and top_left_y < current_y < bottom_right_y:
                # dx = 50 if current_x > self.previous_position_x else -50
                # dy = 30 if current_y > self.previous_position_y else -30
                if current_x > self.previous_position_x + 10:
                    dx = -50
                elif current_x < self.previous_position_x - 10:
                    dx = 50
                else:
                    dx = 0

                if current_y > self.previous_position_y + 10:
                    dy = 50
                elif current_y < self.previous_position_y - 10:
                    dy = -50
                else:
                    dy = 0

                pyautogui.FAILSAFE = False
                pyautogui.moveRel(dx, dy, 0)

                self.previous_position_x = current_x
                self.previous_position_y = current_y
        return image

    def click(self):
        if len(self.list_of_lm):
            if self.fingers[0] == 0 and self.fingers[1] == 1 and self.fingers[2] == 0 and self.fingers[3] == 0 and \
                    self.fingers[4] == 0:

                # autopy.mouse.click()
                pyautogui.click(x=None, y=None, button='left', clicks=1, interval=0.3)
                print("left click")
                # self.wait(.3)

            elif self.fingers[0] == 0 and self.fingers[1] == 0 and self.fingers[2] == 1 and self.fingers[3] == 0 and \
                    self.fingers[4] == 0:
                # autopy.mouse.click(autopy.mouse.Button.RIGHT)
                pyautogui.click(x=None, y=None, button='right', clicks=1, interval=0.3)
                print("right click")
                # self.wait(.3)

            elif self.fingers[0] == 0 and self.fingers[1] == 1 and self.fingers[2] == 1 and self.fingers[3] == 0 and \
                    self.fingers[4] == 1:
                pyautogui.doubleClick(x=None, y=None, interval=0)
                time.sleep(0.3)
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
    def cornerRect(self, img, bbox, length=30, t=5, rt=1,
                   colorR=(255, 0, 255), colorC=(0, 255, 0)):
        """
        :param img: Image to draw on.
        :param bbox: Bounding box [x, y, w, h]
        :param length: length of the corner line
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
        cv2.line(img, (x, y), (x + length, y), colorC, t)
        cv2.line(img, (x, y), (x, y + length), colorC, t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - length, y), colorC, t)
        cv2.line(img, (x1, y), (x1, y + length), colorC, t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + length, y1), colorC, t)
        cv2.line(img, (x, y1), (x, y1 - length), colorC, t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - length, y1), colorC, t)
        cv2.line(img, (x1, y1), (x1, y1 - length), colorC, t)

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
                    # if l < 50:
                    if self.fingers[1] and not self.fingers[2]:
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

    def mode_select(self, image, webcam_width):
        border_space = 20  # Adjust this value to control the space from the border

        # Calculate center coordinates with adjusted space from border
        center_x = int(webcam_width / 2)
        center_y = int(border_space)  # Place the rectangle at the top with border_space

        # Determine rectangle width and height based on desired size and border space
        rectangle_width = 200  # Adjust this value to control the rectangle's width
        rectangle_height = 50  # Adjust this value to control the rectangle's height

        # Calculate top-left corner coordinates
        top_left_x = int(center_x - rectangle_width / 2)
        top_left_y = int(center_y - rectangle_height / 2)

        # Bottom-right corner coordinates (calculated automatically)
        bottom_right_x = top_left_x + rectangle_width
        bottom_right_y = top_left_y + rectangle_height

        # Draw the rectangle with the specified color and filled option
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (160, 114, 0), cv2.FILLED)

        # Text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a suitable font
        text_scale = 0.7  # Adjust this value to control the text size
        text_thickness = 2  # Adjust this value to control the text thickness
        text_color = (255, 255, 255)  # White text color

        # Calculate text position (adjust slightly if needed)
        text_x = int(top_left_x + (rectangle_width - len("mode 1") * text_scale) / 2)
        text_y = int(top_left_y + rectangle_height - text_scale * 5)  # Place the text near the top of the rectangle

        # Add the text "mode 1"
        cv2.putText(image, f"mode {self.mode}", (text_x, text_y), font, text_scale, text_color, text_thickness)

        if self.list_of_lm:
            # print(self.list_of_lm[8][1], self.list_of_lm[8][2])

            if (top_left_x < self.list_of_lm[8][1] < top_left_x + rectangle_width and top_left_y < self.list_of_lm[8][2]
                    < top_left_y + rectangle_height):
                cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (100, 71, 0),
                              cv2.FILLED)
                cv2.putText(image, f"mode {self.mode}", (text_x, text_y), font, text_scale, text_color, text_thickness)

                # when clicked
                if self.fingers[1] and not self.fingers[2]:
                    self.mode = (self.mode + 1) % 4
                    time.sleep(0.5)

        return image
