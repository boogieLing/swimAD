import json
import os
import queue
import shutil
import threading
import logging
import time
from uuid import uuid4
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from associate import alert
from components.drowning_detector import DrowningDetector
from components.event_logger import event_logger
from components.track_lifecycle import g_track_controller
from components.yolo_model import g_yolo_model
from constant import MARGIN_HEIGHT, POOL_HEIGHT, MARGIN_WIDTH, POOL_WIDTH
from components.logger import setup_logger
from entries.track_entry import DetectionEntry, AssociateResult
from swim.trace import MultiViewAssociationStream


class DataAssociate(threading.Thread):
    """
    数据融合主线程：
    - 接收各视角检测（由 FrameDealer 推送到 data_stream）
    - 进行多视角关联，生成主视角结果与 entries
    - 更新全局 tracker（分配 track_id），写入滑窗管理器
    - 执行溺水检测（check_drowning_alerts），并产出事件与推送

    维护要点：
    - 与 YOLO/跟踪器、滑窗管理器之间仅通过数据结构解耦（DetectionEntry/AssociateResult）。
    - 告警链路采用“事件ID”贯穿（accept -> queue -> start/push -> finish）。
    - `alarm_throttle_sec` 控制同一 track 的告警节流窗口。
    """
    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.logger = setup_logger(self.__class__.__name__)  # type: logging.Logger
        self.data_stream = {}
        self.associator = MultiViewAssociationStream()
        self.seen: int = -1
        # self.results: List[Results] = []
        self.object_map = {}
        self.associate_result = queue.Queue(maxsize=10)
        self.alert_queue = queue.Queue(maxsize=100)
        # 告警节流：同一 track 在窗口内重复触发时，抑制重复日志与推送
        self.last_alarm_at: Dict[str, float] = {}
        self.alarm_throttle_sec: float = 10.0

    def add_stream(self, key: str):
        self.logger.info(f"Adding stream for key: {key}")
        self.data_stream[key] = queue.Queue(maxsize=1)

    def stream_data(self, key: str, data: Any):
        # 由 FrameDealer 以固定频率写入检测框/类别/置信度；降级为 debug 防止刷屏
        self.logger.debug(f"Streaming data to key: {key}")
        return self.data_stream[key].put(data)

    def _add_seen(self):
        self.seen += 1

    def stop(self):
        self.logger.info("Stopping DataAssociate thread.")
        self._stop_event.set()
        for key in self.data_stream:  # 空白哨兵，解除阻塞
            try:
                self.data_stream[key].put_nowait(None)
            except queue.Full:
                pass

    def run(self):
        self.logger.info("Starting DataAssociate thread loop.")
        # output_dir = "image/multi_view_detections"
        # if os.path.exists(output_dir):
        #     shutil.rmtree(output_dir)
        while not self._stop_event.is_set():
            self._add_seen()
            try:
                data = []
                for key in self.data_stream:
                    stream_data = self.data_stream[key].get()
                    data.append({
                        'image_path': '',
                        'det': np.array(stream_data, dtype=np.float32),
                    })
                # 多视角关联，输出格式为 [ [bbox, view_name, original_bbox], ... ]
                # print("data: ", data)
                multi_view_detections = self.associator.forward(data)
                # print("multi_view_detections: ", multi_view_detections)

                self.logger.info(f"Associated {len(multi_view_detections)} multi-view detections.")
                # 构建 DetectionEntry 列表
                entries = self._build_detection_entries(multi_view_detections)
                # 构建主视角图像（canvas）并封装为 Results 对象
                main_view_result = self.build_main_view_result(entries)
                # 最终结果写入
                self._update_object_map(entries)
                self._finalize_main_view_result(main_view_result, entries)
                # 更新全局track
                g_track_controller.update_tracker_with_entries(
                    entries=entries, image=main_view_result.orig_img,
                )
                # 用 track.history_observations[-1] 与 entry.key() 精确匹配，关联 track_id
                g_track_controller.fill_detection_entry_track_ids_by_history(entries)
                # visualize_detection_entries(
                #     entries, output_dir=output_dir, output_filename=f"multi_view_result_{self.seen}.jpg"
                # )
                detect_result = {}
                for entry in entries:
                    self.logger.info(f"[Entry] key={entry.key()}, track_id={entry.track_id}")
                    if entry.track_id:
                        g_track_controller.manager.update(entry.track_id, entry, frame_index=self.seen)

                    item = AssociateResult(entry.bbox(), entry.conf, entry.cls_id, entry.track_id)
                    detect_result[entry.track_id] = item

                # 存储当前融合结果
                if self.associate_result.full():
                    self.associate_result.get()
                self.associate_result.put(detect_result)

                self.check_drowning_alerts(detect_result)
            except Exception as e:
                self.logger.error(f"Error in data association loop: {e}", exc_info=True)
        self.logger.info(f"DataAssociate thread loop exited, total: {self.seen}.")

    def check_drowning_alerts(self, detect_result: Dict[str, AssociateResult]) -> None:
        """
        执行溺水检测：
        - 利用 TrackWindowManager 的 `detect_abnormal_tracks` 对所有活跃轨迹执行规则评估
        - `DrowningDetector` 内部在触发时输出汇总 WARNING（包含规则与数值），此处不重复逐条规则日志
        - 告警节流：同一 track 在 `alarm_throttle_sec` 内只受理一次
        - 形成事件（event_id）并入队/推送，事件信息写入 `EventLogger`
        """
        detector = DrowningDetector(use_original_bbox=False)
        abnormal = g_track_controller.manager.detect_abnormal_tracks(detector.as_hook_fn())

        alert_record: list[AssociateResult]= []
        now_ts = time.time()
        accepted_track_ids = []
        event_id: Optional[str] = None
        for track_id, result in abnormal.items():
            if not result or not result.get("alarm", False):
                continue

            # 告警节流：在窗口内重复触发则跳过（仍由 DrowningDetector 输出一次 WARNING 摘要）
            last_ts = self.last_alarm_at.get(track_id, 0)
            if now_ts - last_ts < self.alarm_throttle_sec:
                self.logger.debug(
                    f"[ALARM-SUPPRESS] track_id={track_id} {now_ts - last_ts:.1f}s < {self.alarm_throttle_sec}s")
                continue
            self.last_alarm_at[track_id] = now_ts

            # 仅记录简要接受信息，避免与 DrowningDetector 摘要重复
            rule_cnt = len(result.get("triggered_rules", []))
            if event_id is None:
                event_id = f"D-{int(now_ts*1000)}-{uuid4().hex[:6]}"
            self.logger.info(f"[EVENT-ACCEPT] event_id={event_id} track_id={track_id} rules={rule_cnt}")
            # 事件日志（受理）：只记录规则编号与名称，避免日志过大
            rule_ids = [r.get('rule_id') for r in result.get('triggered_rules', [])]
            rule_names = [r.get('rule_name') for r in result.get('triggered_rules', [])]
            rules = [
                {
                    'id': r.get('rule_id'),
                    'name': r.get('rule_name'),
                    'value': g_track_controller.manager.format_rule_value(r.get('value')),
                }
                for r in result.get('triggered_rules', [])
            ]
            event_logger.log(
                'accept',
                event_id=event_id,
                track_id=track_id,
                rule_count=rule_cnt,
                rule_ids=rule_ids,
                rule_names=rule_names,
                rules=rules,
            )

            # 入队准备推送
            track_item = detect_result.get(track_id)
            if track_item is not None:
                alert_record.append(track_item)
                accepted_track_ids.append(track_id)

        if len(alert_record) > 0:
            if self.alert_queue.full():
                self.alert_queue.get()
            
            self.alert_queue.put(alert_record)
            self.logger.info(f"[EVENT-QUEUE] event_id={event_id} queued={len(alert_record)} tracks: {accepted_track_ids}")
            event_logger.log(
                'queue',
                event_id=event_id,
                track_ids=accepted_track_ids,
                count=len(alert_record),
            )
            # TODO: 一个广播式的异常通知接口
            task = threading.Thread(target=alert.risk_alert, kwargs={'message':alert_record, 'event_id': event_id, 'track_ids': accepted_track_ids})
            task.start()
            #alert.risk_alert(alert_record)

    def associate_multi_view_detections(self, detections: List[Dict[str, Any]]):
        try:
            associated = self.associator.forward(detections)
            self.logger.info(f"Associated {len(associated)} multi-view detections.")
            return associated
        except Exception as e:
            self.logger.error(f"Multi-view association failed: {e}", exc_info=True)
            return []

    def build_main_view_result(self, entries: List[DetectionEntry]) -> Results:
        """
        Build the main view result canvas with boxes for YOLO tracker.

        Args:
            entries: List of DetectionEntry objects

        Returns:
            Results object with image, path, and detection tensor
        """
        det_array = np.array([entry.box_data() for entry in entries])
        if len(det_array) == 0:
            self.logger.info("Empty detection array. Filling with shape (0, 6).")
            det_array = np.zeros((0, 6))

        canvas = np.zeros((2 * MARGIN_HEIGHT + POOL_HEIGHT, 2 * MARGIN_WIDTH + POOL_WIDTH, 3), dtype=np.uint8)
        return Results(
            canvas,
            path=f'main_view_{self.seen:05d}.jpg',
            names=g_yolo_model.names,
            boxes=torch.as_tensor(det_array)
        )

    def _update_object_map(self, entries: List[DetectionEntry]) -> None:
        # 构建 frame_object_map：key = bbox_str，value = (view_name, original_bbox)
        frame_object_map = {}
        for entry in entries:
            key = entry.key()
            if key in frame_object_map:
                self.logger.warning(f"Duplicate entry key: {key}")
            frame_object_map[key] = (entry.view_name, entry.original_bbox)

        self.object_map.update(frame_object_map)
        self.logger.info(f"Updated object_map with {len(frame_object_map)} entries.")

    def _finalize_main_view_result(
            self, result, entries: List[DetectionEntry], is_obb: bool = False
    ) -> None:
        """
        对主视角结果进行最终封装（tensor赋值 + 写入 predictor.results）
        """
        # self.results.append(result)

        det_array = np.array([entry.box_data() for entry in entries])

        # 修复：确保空检测时有正确的形状
        if len(det_array) == 0:
            self.logger.info("Empty detection array in finalize. Filling with shape (0, 6).")
            det_array = np.zeros((0, 6))

        update_tensor = torch.as_tensor(det_array)

        key = "obb" if is_obb else "boxes"
        result.update(**{key: update_tensor})

    @staticmethod
    def _build_detection_entries(detections: List[List[Any]]) -> List[DetectionEntry]:
        """
        构造 DetectionEntry

        Args:
            detections: List of (projected_box, view_name, original_bbox)

        Returns:
            List[DetectionEntry]
        """
        return [
            DetectionEntry(
                box_data=box_data,
                view_name=view_name,
                original_bbox=original_bbox
            )
            for box_data, view_name, original_bbox in detections
        ]

    def get_associate_result(self):
        return self.associate_result.get()
    
    def get_alert(self):
        return self.alert_queue.get()


def visualize_detection_entries(
        entries: List[DetectionEntry],
        output_dir: str = "image/multi_view_detections",
        output_filename: str = "multi_view_result.jpg"
):
    """
    可视化 DetectionEntry 列表，标注 track_id、坐标、cls_id。
    图像尺寸为 POOL_WIDTH*2 × POOL_HEIGHT*2，输出前清空目录。
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: 创建空白图像，放大2倍
    blank_image = np.zeros((POOL_HEIGHT*2, POOL_WIDTH*2, 3), dtype=np.uint8)

    # Step 3: 定义颜色映射 BGR
    COLORS = [
        (0, 0, 255),  # 红色 cls_id=0
        (0, 255, 0),  # 绿色 cls_id=1
        (0, 255, 255),  # 黄色 cls_id=2
        (255, 0, 0),  # 蓝色 cls_id=3
        (255, 0, 255),  # 洋红 cls_id=4
        (255, 255, 0),  # 青色 cls_id=5
    ]
    default_color = (0, 0, 0)

    # Step 4: 遍历绘制
    for entry in entries:
        cls_id = int(entry.cls_id)
        color = COLORS[cls_id] if cls_id < len(COLORS) else default_color

        # 放大坐标
        x1, y1, x2, y2 = map(lambda x: int(x), entry.bbox())
        cv2.rectangle(blank_image, (x1, y1), (x2, y2), color, 2)

        # 构建 label：track_id + 坐标 + cls_id
        tid = entry.track_id if entry.track_id is not None else "None"
        label = f"track:{tid} (({x1},{y1}), ({x2},{y2})) cls:{cls_id}"
        cv2.putText(blank_image, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Step 5: 保存
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, blank_image)
    print(f"[INFO] Detection entries image saved to {output_path}")
