import cv2
import logging
import time
import threading
from queue import Empty, Full, Queue

class StreamHandler:

    def __init__(self, rtsp_source_addr: str, logger: logging.Logger):
        self.logger = logger
        self.stop_event = threading.Event()
        self.rtsp_source_addr: str = rtsp_source_addr
        self.cap = cv2.VideoCapture(rtsp_source_addr, cv2.CAP_FFMPEG)
        self.frame_buffer = Queue(maxsize=24)

        if not self.cap.isOpened():
            self.logger.error(f"Failed to open RTSP stream at address '{rtsp_source_addr}'")
            raise Exception(f"Failed to open RTSP stream to {rtsp_source_addr} with CV2")


    def play(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        try:
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    self.logger.warning("failed to read returned frame")
                    self.stop_event.wait(0.05)
                    continue

                try:
                    self.frame_buffer.put_nowait(frame)
                except Full:
                    try:
                        self.frame_buffer.get_nowait()
                    except Empty:
                        pass
                    self.frame_buffer.put_nowait(frame)
        finally:
            self.cap.release()

    def pause(self):
        self.stop_event.set()
        self.thread.join()
