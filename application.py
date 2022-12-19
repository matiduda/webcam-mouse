import cv2
import datetime
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# global variables
stop_thread = False             # controls thread execution
img = None                      # stores the image retrieved by the camera


def start_capture_thread(cap):
    global img, stop_thread

    # continuously read fames from the camera
    while True:
        _, img = cap.read()

        if (stop_thread):
            break


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

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


while True:
    # initialize time and frame count variables
    last_time = datetime.datetime.now()
    frames = 0

    # Read each frame from the webcam
    # blocks until the entire frame is read
    success, frame = cap.read()

    # --- Processing ---
    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)

    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 50, 50), 2, cv2.LINE_AA)

    # --- FPS ---
    frames += 1

    # compute fps: current_time - last_time
    delta_time = datetime.datetime.now() - last_time
    elapsed_time = delta_time.total_seconds()
    cur_fps = np.around(frames / elapsed_time, 1)

    # draw FPS text and display image
    x, y, w, h = 0, 0, 140, 50

    # FPS Rectangle warning color
    max_fps = 29
    color_diff = np.interp(cur_fps, [0, max_fps], [255, 0])

    # Draw black background rectangle
    cv2.rectangle(frame, (x, x), (x + w, y + h),
                  (0, 0, color_diff), -1)

    # Add text
    cv2.putText(frame, f"FPS: {str(cur_fps)}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.putText(frame, "Press ESC to exit", (int(cap_width - 160), int(cap_height - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Gesture control", frame)

    # wait 1ms for ESC to be pressed
    key = cv2.waitKey(1)
    if (key == 27):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
