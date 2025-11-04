import os
import sys
from pathlib import Path

from ultralytics import YOLO

from constant import YOLO_MODEL

if getattr(sys, 'frozen', False):
    # 打包后的路径
    BASE_PATH = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
else:
    # 开发时的路径
    BASE_PATH = Path(sys.argv[0]).resolve().parent

g_yolo_model = YOLO(Path(BASE_PATH) / YOLO_MODEL).to('cuda')
