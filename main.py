import cv2
import torch
import logging
from yolov5 import YOLOv5

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def detect_people_in_rtsp(rtsp_url, model_name='yolov5s'):
    # Подключение к RTSP потоку
    cap = cv2.VideoCapture(rtsp_url)

    # Проверка, что поток доступен
    if not cap.isOpened():
        logger.error("Ошибка: не удалось подключиться к RTSP потоку.")
        return

    # Загрузка модели YOLOv5 для детекции людей
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', model_name, device=device)

    logger.info("Модель загружена и готова к детекции")

    # Чтение и обработка кадров
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Предупреждение: не удалось получить кадр из видеопотока.")
                break

            # Детектируем объекты на кадре
            results = model(frame)

            # Фильтрация объектов по категории "person"
            for result in results.xyxy[0]:  # Результаты для первого кадра
                x1, y1, x2, y2, conf, cls = result
                if int(cls) == 0:  # 0 - это класс "человек"
                    # Рисуем прямоугольник вокруг человека
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Отображаем кадр с детекцией
            cv2.imshow('Person Detection', frame)

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
