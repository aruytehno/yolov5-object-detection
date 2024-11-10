import cv2
import torch
from yolov5 import YOLOv5


def detect_people_in_rtsp(rtsp_url):
    # Подключение к RTSP потоку
    cap = cv2.VideoCapture(rtsp_url)

    # Загрузка модели YOLOv5 для детекции людей
    model = YOLOv5.load("path/to/yolov5s.pt")

    # Проверка, что поток доступен
    if not cap.isOpened():
        print("Ошибка: не удалось подключиться к RTSP потоку.")
        return

    # Чтение и обработка кадров
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр из видеопотока.")
            break

        # Детектируем объекты на кадре
        results = model(frame)

        # Фильтрация объектов по категории "person"
        for result in results.xyxy[0]:  # Результаты для первого кадра
            x1, y1, x2, y2, conf, cls = result
            if int(cls) == 0:  # 0 - это класс "человек"
                # Рисуем прямоугольник вокруг человека
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Отображаем кадр с детекцией
        cv2.imshow('Person Detection', frame)

        # Выход из цикла при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Закрываем соединение
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Укажите URL RTSP потока
    rtsp_url = 'rtsp://fake.kerberos.io/stream'

    # Запускаем детекцию
    detect_people_in_rtsp(rtsp_url)
