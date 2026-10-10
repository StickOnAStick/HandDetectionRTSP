import logging
from logging import Logger
from app.env import CVStreamAppEnv
from video.stream_handler import StreamHandler

class CVStreamApp:
    _instance = None
    logger: Logger

    def __new__(cls) -> 'CVStreamApp':
        if cls._instance is None:
            # Create intance iff doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.env = CVStreamAppEnv.load()

        logging.basicConfig(
            level=logging.getLevelNamesMapping()[self.env.LOG_LEVEL], 
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        self.stream = StreamHandler(self.env.RTSP_STREAM_SOURCE, self.logger)
        self.stream.play()

        return

    def run(self):
        print("Hello from app!")