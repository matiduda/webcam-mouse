import time
from asyncio import Future

import cv2
import datetime
import numpy as np
import mediapipe as mp

from MouseController import MouseController
from concurrent.futures import ThreadPoolExecutor

from WebcamVideoStream import VideoGet
from FPS import CountsPerSec


def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame


ENVIRONMENT_DEBUG = False
CLICK_DISTANCE = 20
CLICK_COOLDOWN_IN_FRAMES = 10

# initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# create display window
cv2.namedWindow("Gesture control", cv2.WINDOW_NORMAL)

cooldown = 2
frame_count = 0

with ThreadPoolExecutor() as executor:
    old_cursor_pressed = False
    mouse_controller = MouseController(0, ENVIRONMENT_DEBUG)
    move_future = executor.submit(time.sleep(0.0001))
    video_getter = VideoGet(0).start()
    cps = CountsPerSec().start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        # initialize time and frame count variables
        start_time = datetime.datetime.now()
        frames = 0

        # Read each frame from the webcam blocks until the entire frame is read
        frame = video_getter.read()
        frame = cv2.flip(frame, 1)

        # --- Processing ---
        x, y, c = frame.shape

        # Get hand landmark prediction
        results = hands.process(frame)

        # post process the result
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # Mouse click vector
            index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * video_getter.getWidth()
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * video_getter.getHeight()
            thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * video_getter.getWidth()
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * video_getter.getHeight()

            line_color = (0, 255, 0)
            if cooldown > 0:
                line_color = (255, 0, 0)
                cooldown -= 1

            line_thickness = 2
            cv2.line(frame, (int(thumb_x), int(thumb_y)), (int(index_x), int(index_y)),
                     line_color, thickness=line_thickness)

            click_distance = np.sqrt(((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2))

            # COMENTED CODE IS FOR CLICK WITHOUT DRAG AND DROP
            # if click_distance <= CLICK_DISTANCE and cooldown == 0:
            #     mouse_controller.left_click()
            #     time.sleep(0.2)
            #     cooldown = CLICK_COOLDOWN_IN_FRAMES
            # else:
            #     print("mouse released")

            cur_cursor_pressed = False
            if click_distance <= CLICK_DISTANCE:
                cur_cursor_pressed = True
            if old_cursor_pressed != cur_cursor_pressed:
                if cur_cursor_pressed:
                    mouse_controller.left_press()
                else:
                    mouse_controller.left_release()
                old_cursor_pressed = cur_cursor_pressed

            # Mouse position vector
            center_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * video_getter.getWidth()
            center_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * video_getter.getHeight()
            distance_x = (int(video_getter.getWidth() // 2) - center_x)
            distance_y = int(video_getter.getHeight() // 2) - center_y
            distance = np.sqrt(sum(map(lambda x, y: (x - y) ** 2, (distance_x, center_x), (distance_y, center_y))))
            if distance > 50:
                mouse_speed = np.log(distance)
                executor.submit(mouse_controller.move, (-distance_x * mouse_speed) // 50,
                                (-distance_y * mouse_speed) // 50, 0.02)

            # white tracing line
            cv2.line(frame, (int(video_getter.getWidth() // 2), int(video_getter.getHeight() // 2)),
                     (int(center_x), int(center_y)),
                     (255, 255, 255), thickness=1)
            # Printing stats
            if frame_count % 30 == 0:
                frame_count = 0
                # print(f'Mouse speed: {mouse_speed}')

        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Gesture control", frame)
        cps.increment()

# release the webcam and destroy all active windows
cv2.destroyAllWindows()
