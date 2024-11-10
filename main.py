import cv2
import torch
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def detect_people_in_rtsp(rtsp_url, model_name='yolov5s'):
    # Подключение к RTSP потоку
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)  # Попробуйте CAP_GSTREAMER или без указания метода

    # Проверка, что поток доступен
    if not cap.isOpened():
        logger.error("Ошибка: не удалось подключиться к RTSP потоку.")
        return

    # Загрузка модели YOLOv5 для детекции людей
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', model_name, device=device)

    logger.info("Модель загружена и готова к детекции")

    # Уменьшаем разрешение кадра для ускорения обработки
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Уменьшаем размер буфера

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Не удалось получить кадр из видеопотока.")
                break

            # Уменьшаем разрешение кадра для ускорения обработки
            # frame_resized = cv2.resize(frame, (640, 480))  # Уменьшаем размер кадра
            frame_resized = frame  # Используем кадр без изменения разрешения

            # Детектируем объекты на кадре
            results = model(frame_resized)

            # Фильтрация объектов по категории "person"
            for result in results.xyxy[0]:  # Результаты для первого кадра
                x1, y1, x2, y2, conf, cls = result
                if int(cls) == 0:  # 0 - это класс "человек"
                    # Рисуем прямоугольник вокруг человека
                    cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame_resized, f"Person {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Отображаем кадр с детекцией
            cv2.imshow('Person Detection', frame_resized)

            # Выход из цикла при нажатии клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Завершение работы по запросу пользователя (нажата клавиша 'q')")
                break
    except Exception as e:
        logger.error(f"Ошибка во время обработки кадров: {e}")
    finally:
        # Закрываем соединение
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Соединение с RTSP потоком закрыто")


if __name__ == "__main__":
    # Укажите URL RTSP потока
    rtsp_url = 'rtsp://fake.kerberos.io/stream'

    # Запускаем детекцию
    detect_people_in_rtsp(rtsp_url)

