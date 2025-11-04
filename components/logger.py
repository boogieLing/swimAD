import logging
import os
from datetime import datetime


def setup_logger(name: str, log_dir: str = './logs') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 日期后缀（格式如：2025-07-31）
        date_str = datetime.now().strftime('%Y-%m-%d')
        log_file = os.path.join(log_dir, f"{name}_{date_str}.log")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
