import json
import os
import threading
import time
from queue import Queue, Empty
from typing import Optional, Dict, Any


class EventLogger(threading.Thread):
    """
    轻量级“事件日志写入器”（JSON Lines），用于溺水告警“可溯源”的链路记录。

    设计说明（给维护者）：
    - 输出介质：JSONL（每行一个 JSON），默认路径为 `./logs/alert_events.jsonl`。
    - 写入方式：异步队列 + 守护线程，尽量避免影响主线程耗时。
    - 容错策略：单条写入失败被忽略，不会中断主流程；可根据需要改为告警上报。
    - 线程生命周期：模块级单例在 import 时启动；如需优雅停机，调用 `stop()` 即可。
    """

    def __init__(self, file_path: str = './logs/alert_events.jsonl'):
        super().__init__(daemon=True)
        self._queue: Queue = Queue()
        self._running = True
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def push(self, item: Dict[str, Any]) -> None:
        """直接压入一条结构化事件。"""
        self._queue.put(item)

    def log(self, event: str, **kwargs) -> None:
        """快速构造并压入事件（自动附加时间戳与事件名）。"""
        item = {
            'time': time.time(),
            'event': event,
        }
        item.update(kwargs)
        self.push(item)

    def run(self) -> None:
        """线程主循环：逐条从队列读取并写入文件。"""
        with open(self.file_path, 'a', encoding='utf-8') as f:
            while self._running:
                try:
                    item = self._queue.get(timeout=1.0)
                except Empty:
                    continue
                try:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()
                except Exception:
                    # 忽略单条写入错误，避免影响主流程
                    pass

    def stop(self) -> None:
        """请求停止线程（不会立刻退出，等待下一次循环）。"""
        self._running = False


# 模块级单例，随模块导入即启动
event_logger = EventLogger()
event_logger.start()
