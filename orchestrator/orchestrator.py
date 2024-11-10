import logging
from runner.runner import Runner

logger = logging.getLogger(__name__)

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
