import logging
import os
import tempfile
import shutil

# 指定文件保存的本地路径
KEYWORD_FILE = 'output/keyword'
DATA_FILE = 'output/data'

# 是否显示详细日志
log_verbose = True

# 日志格式
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)


# 日志存储路径
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

# 临时文件目录，主要用于文件对话
BASE_TEMP_DIR = os.path.join(tempfile.gettempdir(), "tmpfile")
try:
    shutil.rmtree(BASE_TEMP_DIR)
except Exception:
    pass
os.makedirs(BASE_TEMP_DIR, exist_ok=True)

