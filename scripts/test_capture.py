import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.capture.video_capture import VideoSource, FPSMeter

# True  - видео идёт в темпе оригинала (для просмотра)
# False - без пауз, fps показывает максимальную скорость цикла
REALTIME = True


def main():
    video_path = "C:/Users/ekose/PycharmProjects/big_data_analysis_1/data/street.mp4"
    # video_path = 0

    source = VideoSource(source=video_path, reconnect_sec=2.0)
    fps_meter = FPSMeter()

    window_name = "Video"

    try:
        source.open()

        video_fps = source.cap.get(cv2.CAP_PROP_FPS) or 30
        frame_time = 1.0 / video_fps
        next_time = time.perf_counter()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 450)

        while True:
            frame = source.read()
            fps = fps_meter.tick()

            # если видео очень большое, сжимаем его
            height, width = frame.shape[:2]
            if width > 800:
                new_width = 800
                new_height = int(height * (new_width / width))
                frame = cv2.resize(frame, (new_width, new_height))

            text = f"FPS: {fps:.1f}"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # выдерживаем темп оригинала по расписанию, а не через waitKey
            if REALTIME:
                next_time += frame_time
                delay = next_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_time = time.perf_counter()

            if key == ord('q'):
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()