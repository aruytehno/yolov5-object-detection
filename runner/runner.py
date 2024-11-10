import cv2
import logging
from inference.detector import Detector

logger = logging.getLogger(__name__)

class Runner:
    def __init__(self, rtsp_url, active):
        self.rtsp_url = rtsp_url
        self.active = active
        self.detector = Detector()

    def process_video_stream(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("Ошибка: не удалось подключиться к RTSP потоку.")
            return

        try:
            while self.active.value:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Не удалось получить кадр из видеопотока.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.detector.detect(frame_rgb)

                for result in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = result
                    if int(cls) == 0:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow('Person Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Завершение работы по запросу пользователя (нажата клавиша 'q')")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Соединение с RTSP потоком закрыто")

def start_runner(rtsp_url, active):
    runner = Runner(rtsp_url, active)
    runner.process_video_stream()
