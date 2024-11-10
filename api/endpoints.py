from fastapi import APIRouter, HTTPException
from api.models import ScenarioState
from multiprocessing import Value, Process
import cv2
from runner.runner import start_runner  # Изменили импорт

router = APIRouter()
active = Value('b', False)
process = None

@router.get("/health")
def health_check():
    return {"status": "healthy"}

@router.get("/status")
def status_check():
    if not active.value:
        return {"status": "inactive", "details": "Система остановлена"}

    cap = cv2.VideoCapture('rtsp://fake.kerberos.io/stream', cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Ошибка подключения к RTSP потоку")
    cap.release()

    return {"status": "active", "details": "Система работает корректно"}

@router.get("/scenario/{scenario_id}")
def get_scenario_status(scenario_id: int):
    state = "started" if active.value else "stopped"
    return {"scenario_id": scenario_id, "state": state}

@router.post("/scenario/{scenario_id}/state")
def change_scenario_state(scenario_id: int, state: ScenarioState):
    global process
    if state.action == "start":
        if not active.value:
            active.value = True
            process = Process(target=start_runner, args=('rtsp://fake.kerberos.io/stream', active))
            process.start()
            return {"scenario_id": scenario_id, "state": "started"}
        else:
            return {"error": "Сценарий уже запущен"}
    elif state.action == "stop":
        if active.value:
            active.value = False
            process.join()
            return {"scenario_id": scenario_id, "state": "stopped"}
        else:
            return {"error": "Сценарий не запущен"}
    else:
        return {"error": "Недопустимое действие. Используйте 'start' или 'stop'."}
