import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.capture.video_capture import VideoSource
from src.monitoring.fps_meter import FPSMeter
from src.inference.detector import Detector


def main():
    #video_path = 0
    video_path = "C:/Users/ekose/PycharmProjects/big_data_analysis_1/data/car.mp4"

    model_path = "models/yolo11n.pt"

    source = VideoSource(source=video_path, reconnect_sec=2.0)
    fps_meter = FPSMeter()
    detector = Detector(model_path=model_path, confidence=0.5)

    window_name = "YOLO Detection"

    try:
        source.open()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 450)

        while True:
            frame = source.read()
            if frame is None:
                continue

            fps = fps_meter.tick()

            height, width = frame.shape[:2]
            if width > 800:
                new_width = 800
                new_height = int(height * (new_width / width))
                frame = cv2.resize(frame, (new_width, new_height))

            detections = detector.predict(frame)

            # отрисовка
            for det in detections:
                # прямоугольник вокруг объекта
                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)

                # класс и уверенность
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(frame, label, (det.x1, det.y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)

            text = f"FPS: {fps:.1f}"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()