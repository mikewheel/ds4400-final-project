from contextlib import suppress
from datetime import datetime
from logging import Formatter, getLogger, FileHandler, StreamHandler, INFO, DEBUG
from os import mkdir
from pathlib import Path
from sys import stdout

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INPUT_DATA_DIR = DATA_DIR / "input"
OUTPUT_DATA_DIR = DATA_DIR / "output"
VISUALIZATION_DATA_DIR = DATA_DIR / "visualizations"

with suppress(FileExistsError):
    mkdir(DATA_DIR)

with suppress(FileExistsError):
    mkdir(INPUT_DATA_DIR)
    
with suppress(FileExistsError):
    mkdir(OUTPUT_DATA_DIR)
    
with suppress(FileExistsError):
    mkdir(VISUALIZATION_DATA_DIR)

# Descriptions of the basis function expansions
BFE_DESCS = ["base", "fixed_acidity_removed", "volatine_acidity_removed", "citric_acid_removed",
             "residual_sugar_removed", "chlorides_removed", "free_sulfur_dioxide_removed",
             "total_sulfur_dioxide_removed", "density_removed", "pH_removed", "suplhates_removed",
             "alcohol_removed", "fixed_acidity_squared", "volatine_acidity_squared",
             "citric_acid_squared", "residual_sugar_squared", "chlorides_squared", "free_sulfur_dioxide_squared",
             "total_sulfur_dioxide_squared", "density_squared", "pH_squared", "suplhates_squared", "alcohol_squared"]


def make_logger(module_name):
    logger = getLogger(module_name)
    logger.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    
    file_handler = FileHandler(OUTPUT_DATA_DIR / f'training_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log',
                               mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(DEBUG)
    
    stdout_handler = StreamHandler(stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(INFO)
    
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger
