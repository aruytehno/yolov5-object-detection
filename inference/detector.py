import torch
import logging

logger = logging.getLogger(__name__)

class Detector:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)
        logger.info("Модель загружена и готова к детекции")

    def detect(self, frame):
        return self.model(frame)
