import cv2
import datetime
import numpy as np
import mediapipe as mp

from MouseController import MouseController
from concurrent.futures import ThreadPoolExecutor

ENVIRONMENT_DEBUG = True
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

# initialize webcam capture object
cap = cv2.VideoCapture(0)

# --- FPS ---
# retrieve properties of the capture object
cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap_fps = cap.get(cv2.CAP_PROP_FPS)
fps_sleep = int(1000 / cap_fps)
print('* Capture width:', cap_width)
print('* Capture height:', cap_height)
print('* Capture FPS:', cap_fps, 'ideal wait time between frames:', fps_sleep, 'ms')

cooldown = 0
frame_count = 0

mouse_controller = MouseController(0, ENVIRONMENT_DEBUG)



def update_fps(frames):
    global x, y
    # --- FPS ---
    frames += 1
    # compute fps: current_time - last_time
    delta_time = datetime.datetime.now() - start_time
    elapsed_time = delta_time.total_seconds()
    cur_fps = np.around(frames / elapsed_time, 1)
    # draw FPS text and display image
    x, y, w, h = 0, 0, 140, 50
    # FPS Rectangle warning color
    max_fps = 29
    color_diff = np.interp(cur_fps, [0, max_fps], [255, 0])
    # Draw FPS background rectangle
    cv2.rectangle(frame, (x, x), (x + w, y + h), (0, 0, color_diff), -1)
    # Add text
    cv2.putText(frame, f"FPS: {str(cur_fps)}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return frames


def draw_static_frame_data():
    cv2.putText(frame, "Press ESC to exit", (int(cap_width - 160), int(cap_height - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Gesture control", frame)


with ThreadPoolExecutor() as executor:
    while True:
        # initialize time and frame count variables
        start_time = datetime.datetime.now()
        frames = 0

        # Read each frame from the webcam blocks until the entire frame is read
        success, frame = cap.read()

        # --- Processing ---
        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)

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
            index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * cap_width
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * cap_height
            thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * cap_width
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * cap_height

            line_color = (0, 255, 0)
            if cooldown > 0:
                line_color = (255, 0, 0)
                cooldown -= 1

            line_thickness = 2
            cv2.line(frame, (int(thumb_x), int(thumb_y)), (int(index_x), int(index_y)),
                     line_color, thickness=line_thickness)

            click_distance = np.sqrt(
                ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2))
            if click_distance <= CLICK_DISTANCE and cooldown == 0:
                executor.submit(mouse_controller.left_click())
                cooldown = CLICK_COOLDOWN_IN_FRAMES

            # Mouse position vector
            center_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * cap_width
            center_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * cap_height
            distance_x = (int(cap_width // 2) - center_x)
            distance_y = int(cap_height // 2) - center_y
            if abs(distance_x) > 40 or abs(distance_y) > 40:
                executor.submit(mouse_controller.move, -distance_x // 6, -distance_y // 6,
                                (datetime.datetime.now() - start_time).total_seconds())

            # mouse_speed = np.log(center_distance)
            # white tracing line
            cv2.line(frame, (int(cap_width // 2), int(cap_height // 2)), (int(center_x), int(center_y)),
                     (255, 255, 255), thickness=1)
            # Printing stats
            if frame_count % 30 == 0:
                frame_count = 0
                # print(f'Mouse speed: {mouse_speed}')

        frames = update_fps(frames)
        draw_static_frame_data()

        # wait 1ms for ESC to be pressed
        key = cv2.waitKey(1)
        if key == 27:
            break

        frame_count += 1

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
