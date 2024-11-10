# import cv2
# import torch
# import logging
# import time
# from fastapi import FastAPI
# from pydantic import BaseModel
# import uvicorn
# from fastapi import HTTPException
# from multiprocessing import Process, Value
#
# # Настройка логирования
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)
#
# # FastAPI приложение
# app = FastAPI()
#
#
# class ScenarioState(BaseModel):
#     action: str  # "start" or "stop"
#
# class Orchestrator:
#     def __init__(self, rtsp_url):
#         self.rtsp_url = rtsp_url
#         self.runner = Runner(rtsp_url)
#         self.state_machine = StateMachine()
#
#     def start(self):
#         logger.info("Запуск оркестратора")
#         self.state_machine.transition_to('init_startup')
#         self.state_machine.transition_to('in_startup_processing')
#
#         logger.info("Инициализация системы")
#         self.runner.start()
#         self.state_machine.transition_to('active')
#         logger.info("Система активирована")
#
#     def stop(self):
#         logger.info("Остановка оркестратора")
#         self.state_machine.transition_to('init_shutdown')
#         self.state_machine.transition_to('in_shutdown_processing')
#
#         self.runner.stop()
#         self.state_machine.transition_to('inactive')
#         logger.info("Система остановлена")
#
# class StateMachine:
#     def __init__(self):
#         self.state = 'inactive'
#
#     def transition_to(self, new_state):
#         logger.info(f"Переход в состояние: {new_state}")
#         self.state = new_state
#
# class Runner:
#     def __init__(self, rtsp_url, active):
#         self.rtsp_url = rtsp_url
#         self.active = active
#
#     def process_video_stream(self):
#         # Подключение к RTSP потоку
#         cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
#         if not cap.isOpened():
#             logger.error("Ошибка: не удалось подключиться к RTSP потоку.")
#             return
#
#         # Загрузка модели YOLOv5
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)
#         logger.info("Модель загружена и готова к детекции")
#
#         # Обработка потока
#         try:
#             while self.active.value:
#                 ret, frame = cap.read()
#                 if not ret:
#                     logger.warning("Не удалось получить кадр из видеопотока.")
#                     break
#
#                 # Преобразование BGR в RGB
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#                 # Детектируем объекты на кадре
#                 results = model(frame_rgb)
#
#                 # Фильтрация объектов по категории "person"
#                 for result in results.xyxy[0]:
#                     x1, y1, x2, y2, conf, cls = result
#                     if int(cls) == 0:  # 0 - это класс "человек"
#                         # Рисуем прямоугольник вокруг человека
#                         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                         cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1) - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#                 # Отображаем кадр с детекцией
#                 cv2.imshow('Person Detection', frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     logger.info("Завершение работы по запросу пользователя (нажата клавиша 'q')")
#                     break
#         finally:
#             cap.release()
#             cv2.destroyAllWindows()
#             logger.info("Соединение с RTSP потоком закрыто")
#
#
# def start_runner(rtsp_url, active):
#     runner = Runner(rtsp_url, active)
#     runner.process_video_stream()
#
#
# active = Value('b', False)  # Флаг активности видеопотока
# process = None
#
# @app.get("/health")
# def health_check():
#     """
#     Эндпоинт для проверки работоспособности приложения.
#     Возвращает HTTP статус 200, если приложение работает.
#     """
#     return {"status": "healthy"}
#
#
# @app.get("/status")
# def status_check():
#     """
#     Эндпоинт для проверки состояния системы и подключений.
#     Проверяет:
#     - Состояние флага активности потока
#     - Состояние подключения к RTSP потоку
#     """
#     if not active.value:
#         return {"status": "inactive", "details": "Система остановлена"}
#
#     # Проверяем подключение к RTSP потоку
#     cap = cv2.VideoCapture('rtsp://fake.kerberos.io/stream', cv2.CAP_FFMPEG)
#     if not cap.isOpened():
#         raise HTTPException(status_code=500, detail="Ошибка подключения к RTSP потоку")
#
#     # Закрываем соединение с потоком
#     cap.release()
#
#     return {"status": "active", "details": "Система работает корректно"}
#
# @app.get("/scenario/{scenario_id}")
# def get_scenario_status(scenario_id: int):
#     """
#     Получить информацию о состоянии сценария по его ID.
#     """
#     # Проверяем, запущен ли процесс (активное значение флага `active`).
#     state = "started" if active.value else "stopped"
#     return {"scenario_id": scenario_id, "state": state}
#
#
# @app.post("/scenario/{scenario_id}/state")
# def change_scenario_state(scenario_id: int, state: ScenarioState):
#     global process
#     if state.action == "start":
#         if not active.value:
#             active.value = True
#             process = Process(target=start_runner, args=('rtsp://fake.kerberos.io/stream', active))
#             process.start()
#             return {"scenario_id": scenario_id, "state": "started"}
#         else:
#             return {"error": "Сценарий уже запущен"}
#     elif state.action == "stop":
#         if active.value:
#             active.value = False
#             process.join()
#             return {"scenario_id": scenario_id, "state": "stopped"}
#         else:
#             return {"error": "Сценарий не запущен"}
#     else:
#         return {"error": "Недопустимое действие. Используйте 'start' или 'stop'."}
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import uvicorn
from api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
