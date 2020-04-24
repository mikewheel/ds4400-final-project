from logging import Formatter, getLogger, FileHandler, StreamHandler, INFO, DEBUG
from pathlib import Path
from sys import stdout

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INPUT_DATA_DIR = DATA_DIR / "input"
OUTPUT_DATA_DIR = DATA_DIR / "output"
VISUALIZATION_DATA_DIR = DATA_DIR / "visualizations"


def make_logger(module_name):
    logger = getLogger(module_name)
    logger.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    
    file_handler = FileHandler(OUTPUT_DATA_DIR / "search.log", mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(DEBUG)
    
    stdout_handler = StreamHandler(stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(INFO)
    
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger
