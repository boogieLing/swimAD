import queue
import threading
import time
import cv2
import numpy as np

from associate import associate
from components.logger import setup_logger
from components.yolo_model import g_yolo_model
from constant import FREQUENCY


def now():
    return int(time.time_ns() / 1000000)

class FrameProcessor(threading.Thread):
    def __init__(self, key: str, source: str):
        super().__init__(daemon=True)
        self.source_key = key
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.stop_event = threading.Event()
        self.dealer = FrameDealer(key)

    # def image(self):
    #     a = 1
    #     start = now()
    #     while a < 200:
    #         # start = now()
    #         aa = str(a+382).zfill(4)
    #         results = self.model(f'image/{self.source}/morning_{self.source}_{aa}.jpg', classes=[1, 2])
    #         annotated_frame = results[0].plot()
    #         _, jpeg = cv2.imencode('.jpg', annotated_frame)
    #         self.current_frame = jpeg.tobytes()
    #
    #         boxes = results[0].boxes
    #         xyxy_coords = boxes.xyxy.cpu().numpy()  # 获取绝对坐标（NumPy数组）
    #         confidences = boxes.conf.tolist()  # 获取置信度列表
    #         classes = boxes.cls.tolist()
    #
    #         # 合并为 (13, 3) 的数组（每行: [arr1[i], arr2[i], arr3[i]]）
    #         combined = np.column_stack((xyxy_coords, confidences, classes))
    #         associate.stream_data(self.source, combined)
    #         a += 1
    #
    #         sleep = (500 - (now() - start - 500 * (a-1)))/1000
    #         print("~~~~~~~~~~~~~", sleep)
    #         time.sleep(sleep)

    # 处理本地视频模拟
    def video(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)  # 视频帧率
        split_time = 1000 / fps
        count = int(fps / FREQUENCY) #5
        index = 1

        # print(split_time, fps, count)
        start_time = now()
        num = 0
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue

            if index % count == 0:
                self.dealer.put_frame(frame, timeout=1.0)
                index = 1
            else:
                index += 1

            sleep_time = split_time - (now() - start_time - num * split_time)
            if sleep_time > 0:
                time.sleep(sleep_time / 1000)
            num += 1

    # 处理摄像头视频流
    def stream(self):
        split_time = 1000 / FREQUENCY  # 间隔时间
        last_process_time = now()

        while self.cap.isOpened():
            current_time = now()
            ret, frame = self.cap.read()
            if not ret:
                continue

            if current_time - last_process_time >= split_time:
                last_process_time = current_time
                # 添加超时防止队列阻塞
                self.dealer.put_frame(frame, timeout=1.0)

    def run(self):
        if "rtsp" in self.source:
            self.stream()
        else:
            self.video()

    def get_current_frame(self):
        return self.dealer.current_frame

    def stop(self):
        self.stop_event.set()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'dealer'):
            self.dealer.stop()
            # 等待线程结束
        self.join(timeout=2.0)

class FrameDealer(threading.Thread):
    def __init__(self, key: str):
        super().__init__(daemon=True)
        self.key = key
        self.frame_queue = queue.Queue(maxsize=5)
        # self.model = YOLO(Path(BASE_PATH) / YOLO_MODEL).to('cuda')
        self.model = g_yolo_model
        self.stop_event = threading.Event()
        self.current_frame = None
        self.logger = setup_logger(self.__class__.__name__)

        self.start()

    def run(self):
        num = 0
        while not self.stop_event.is_set():
            try:
                num += 1
                frame = self.frame_queue.get(timeout=1.0)
                if frame is None:  # 停止信号
                    break

                results = self.model(frame, verbose=False, classes=[1, 2])
                annotated_frame = results[0].plot()
                _, jpeg = cv2.imencode('.jpg', annotated_frame)
                self.current_frame = jpeg.tobytes()
                # jpeg.tofile(f"images/vision{self.key}-{num}.jpg")

                # 显式释放内存
                del annotated_frame
                del jpeg

                boxes = results[0].boxes
                if boxes is not None:
                    xyxy_coords = boxes.xyxy.cpu().numpy()  # 获取绝对坐标（NumPy数组）
                    confidences = boxes.conf.tolist()  # 获取置信度列表
                    classes = boxes.cls.tolist()

                    if len(xyxy_coords) > 0:
                        # 合并为 (13, 3) 的数组（每行: [arr1[i], arr2[i], arr3[i]]）
                        combined = np.column_stack((xyxy_coords, confidences, classes))
                        associate.stream_data(self.key, combined)
                    else:
                        associate.stream_data(self.key, [])
                else:
                    associate.stream_data(self.key, [])

                # 释放其他资源
                del results
                self.frame_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in frame dealer: {e}", exc_info=True)

    def put_frame(self, frame, timeout=None):
        try:
            self.frame_queue.put(frame, timeout=timeout, block=False)
            return True
        except queue.Full:
            # 队列已满，丢弃最老的帧
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
            except queue.Empty:
                pass
            # 尝试再次放入新帧
            try:
                self.frame_queue.put(frame, timeout=timeout, block=False)
                return True
            except queue.Full:
                self.logger.warning("帧队列已满，丢弃帧")
                return False

    def stop(self):
        self.stop_event.set()
        # 发送停止信号
        try:
            self.frame_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        self.join(timeout=2.0)

