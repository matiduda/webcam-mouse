from datetime import time
from threading import Thread, Semaphore
import cv2
import time

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream_width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.stream_height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

        (self.grabbed, self.frame) = self.stream.read()
        self.frame = cv2.flip(self.frame, 1)
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

    def getWidth(self):
        return self.stream_width

    def getHeight(self):
        return self.stream_height
