import os

VERSION = 0.1
LUCKY_SEED = 42

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

OBJECTS_DIR = os.path.join(ROOT_DIR, "objects")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DATA_DIR = os.path.join(ROOT_DIR, "data")

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(filename)s | %(lineno)d | %(message)s"
)
LOG_LEVEL = "DEBUG"
LOG_FILE = os.path.join(LOGS_DIR, "threat_text_classification.log")
LOG_FILE_MAX_BYTES = 1048576
LOG_FILE_BACKUP_COUNT = 2
