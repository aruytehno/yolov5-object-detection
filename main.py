import cv2
import torch
import logging
import time
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI приложение
app = FastAPI()

class ScenarioState(BaseModel):
    action: str  # "start" or "stop"

class Orchestrator:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.runner = Runner(rtsp_url)
        self.state_machine = StateMachine()

    def start(self):
        logger.info("Запуск оркестратора")
        self.state_machine.transition_to('init_startup')
        self.state_machine.transition_to('in_startup_processing')

        logger.info("Инициализация системы")
        self.runner.start()
        self.state_machine.transition_to('active')
        logger.info("Система активирована")

    def stop(self):
        logger.info("Остановка оркестратора")
        self.state_machine.transition_to('init_shutdown')
        self.state_machine.transition_to('in_shutdown_processing')

        self.runner.stop()
        self.state_machine.transition_to('inactive')
        logger.info("Система остановлена")

class StateMachine:
    def __init__(self):
        self.state = 'inactive'

    def transition_to(self, new_state):
        logger.info(f"Переход в состояние: {new_state}")
        self.state = new_state

class Runner:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.active = False

    def start(self):
        self.active = True
        logger.info("Запуск runner'а")
        self.process_video_stream()

    def stop(self):
        self.active = False
        logger.info("Остановка runner'а")

    def process_video_stream(self):
        # Подключение к RTSP потоку
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("Ошибка: не удалось подключиться к RTSP потоку.")
            return

        # Загрузка модели YOLOv5
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)
        logger.info("Модель загружена и готова к детекции")

        # Обработка потока
        try:
            while self.active:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Не удалось получить кадр из видеопотока.")
                    break

                # Преобразование BGR в RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Детектируем объекты на кадре
                results = model(frame_rgb)

                # Фильтрация объектов по категории "person"
                for result in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = result
                    if int(cls) == 0:  # 0 - это класс "человек"
                        # Рисуем прямоугольник вокруг человека
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Отображаем кадр с детекцией
                cv2.imshow('Person Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Завершение работы по запросу пользователя (нажата клавиша 'q')")
                    break
        except Exception as e:
            logger.error(f"Ошибка во время обработки кадров: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Соединение с RTSP потоком закрыто")

# Создание глобального экземпляра Orchestrator для локального запуска и API
orchestrator = Orchestrator(rtsp_url='rtsp://fake.kerberos.io/stream')

@app.get("/scenario/{scenario_id}")
def get_scenario_status(scenario_id: int):
    """
    Получить информацию о состоянии сценария по его ID.
    """
    return {"scenario_id": scenario_id, "state": orchestrator.state_machine.state}

@app.post("/scenario/{scenario_id}/state")
def change_scenario_state(scenario_id: int, state: ScenarioState):
    """
    Изменить состояние сценария по его ID.
    Параметры:
      - action: "start" для запуска, "stop" для остановки.
    """
    if state.action == "start":
        orchestrator.start()
    elif state.action == "stop":
        orchestrator.stop()
    else:
        return {"error": "Недопустимое действие. Используйте 'start' или 'stop'."}

    return {"scenario_id": scenario_id, "new_state": orchestrator.state_machine.state}

if __name__ == "__main__":
    # Запуск FastAPI с использованием uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
