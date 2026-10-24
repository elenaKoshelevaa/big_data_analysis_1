from __future__ import annotations
import time
import logging
import cv2
from src.monitoring.fps_meter import FPSMeter

logger = logging.getLogger(__name__)

class VideoSource:
    def __init__(self, source: str | int, reconnect_sec: float = 2.0):
        self.source = source
        self.reconnect_sec = reconnect_sec
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            error_msg = f"Cannot open video source: {self.source}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            self.open()

        ok, frame = self.cap.read()

        if ok and frame is not None:
            return frame

        # если кадр не прочитался (потеря связи)
        logger.warning(f"Stream lost. Reconnecting in {self.reconnect_sec} seconds...")
        time.sleep(self.reconnect_sec)

        try:
            self.open()
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("Video stream is unavailable after reconnect")
            return frame
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
            raise RuntimeError("Video stream is unavailable after reconnect")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            logger.info("Video source closed")