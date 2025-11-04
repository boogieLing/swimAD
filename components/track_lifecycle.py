import logging
import os
import threading
from typing import Optional, List

import numpy as np
import yaml

from components.logger import setup_logger
from components.track_window_manager import TrackWindowManager
from constant import (
    TRACK_LIFECYCLE_TTL,
    TRACK_LIFECYCLE_SNAP_INTERVAL,
    TRACK_LIFECYCLE_WINDOWS_SPAN,
    MODEL_BASE_PATH,
    CONFIG_BASE_PATH,
)
from entries.track_entry import DetectionEntry
from trackers.init_track import init_trackers


class TrackManagerController:
    """
    TrackWindowManager 生命周期托管器（带线程管理、异常安全、上下文封装）
    """

    def __init__(
            self,
            max_window_size: int = 30,
            ttl: int = 10,
            save_dir: str = "track_logs",
            snapshot_interval: int = 60,
            auto_shutdown: bool = True,
            frame_window_span: Optional[int] = None
    ):
        self._manager = TrackWindowManager(
            max_window_size=max_window_size,
            ttl=ttl,
            save_dir=save_dir,
            snapshot_interval=snapshot_interval,
            frame_window_span=frame_window_span,
        )

        self._closed = False
        self._auto_shutdown = auto_shutdown
        self._lock = threading.Lock()
        self.logger = setup_logger(self.__class__.__name__)  # type: logging.Logger
        self.tracker = None
        self._track_init()

    def _track_init(self):
        trackers = init_trackers(
            tracking_method="ocsort",
            reid_model=f"{MODEL_BASE_PATH}/osnet_x0_25_msmt17.pt"
        )  # TODO： init_trackers
        if len(trackers) == 0:
            self.logger.error("No trackers initialized.")
            raise ValueError("No trackers initialized.")
        self.tracker = trackers[0]  # TODO: tracks should be configurable

    @property
    def manager(self) -> TrackWindowManager:
        if self._closed:
            raise RuntimeError("TrackWindowManager is already closed.")
        return self._manager

    @property
    def closed(self) -> bool:
        return self._closed

    def shutdown(self):
        with self._lock:
            if self._closed:
                return
            try:
                self._manager.shutdown()
            except Exception as e:
                print(f"[TrackManagerController] Shutdown error: {e}")
            finally:
                self._closed = True

    def fill_detection_entry_track_ids_by_history(
            self, entries: List[DetectionEntry],
    ) -> None:
        """
        用 tracker 的 history_observations[-1] 构造 key，
        填入 DetectionEntry.track_id

        Args:
            entries: 当前帧的 DetectionEntry 列表
        """
        entry_dict = {entry.key(): entry for entry in entries}
        matched, unmatched = 0, 0
        matched_ids, unmatched_ids = [], []
        for track in self.tracker.active_tracks:
            # self.logger.info(f"fill_detection_entry_track_ids_by_history Track: {track.id}")
            if not track.history_observations:
                self.logger.info(f"Track{track.id}: No history observations.")
                continue
            box = track.history_observations[-1][:4]
            key = f"{int(box[0])}_{int(box[1])}_{int(box[2])}_{int(box[3])}"
            if key in entry_dict:
                entry_dict[key].track_id = track.id
                matched += 1
                matched_ids.append(track.id)
            else:
                unmatched += 1
                unmatched_ids.append(track.id)

        self.logger.info(f"[fill_track_id_by_history] Matched: {matched}, Unmatched: {unmatched}")
        self.logger.info(f"[fill_track_id_by_history] Matched ids: {matched_ids}, Unmatched ids: {unmatched_ids}")

    def update_tracker_with_entries(self, entries: List[DetectionEntry], image: np.ndarray) -> None:
        """
        使用当前帧的 DetectionEntry 更新 tracker 状态（不返回结果）
        """
        try:
            det_array = np.array([entry.box_data() for entry in entries])
            if len(det_array) > 0:
                tracks = self.tracker.update(det_array, image)
                if not isinstance(tracks, (list, tuple)):
                    self.logger.warning(f"Tracker update did not return list/tuple. Got: {type(tracks)}")
                elif len(tracks) == 0:
                    self.logger.info("Tracker update returned empty tracks.")
                else:
                    self.logger.info(f"{len(tracks)} tracks updated.")
                self.logger.info(f"[update_tracker_with_entries] Tracks:\n {tracks}")
            else:
                self.logger.warning("No detections, skipped tracker update.")
        except Exception as e:
            self.logger.error(f"Tracker update failed: {e}", exc_info=True)

    def __enter__(self):
        return self.manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._auto_shutdown:
            self.shutdown()


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_tracker_config() -> dict:
    """
    加载滑动窗口/生命周期配置（YAML，可缺省）。
    优先从 `config/tracker.yaml` 读取；不存在则使用默认值（与当前代码一致）。
    """
    default_cfg = {
        'max_window_size': 30,
        'ttl': TRACK_LIFECYCLE_TTL,
        'save_dir': 'track_logs',
        'snapshot_interval': TRACK_LIFECYCLE_SNAP_INTERVAL,
        'frame_window_span': TRACK_LIFECYCLE_WINDOWS_SPAN,
    }
    path = os.path.join(CONFIG_BASE_PATH, 'tracker.yaml')
    if not os.path.isfile(path):
        return default_cfg
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            cfg = data.get('tracker', data)
            return _deep_merge(default_cfg, cfg)
    except Exception as e:
        print(f"[TrackManagerController] 加载 tracker 配置失败，使用默认。error={e}")
        return default_cfg


# 从配置文件初始化全局 Track 管理器
_tracker_cfg = _load_tracker_config()
g_track_controller = TrackManagerController(
    max_window_size=_tracker_cfg.get('max_window_size', 30),
    ttl=_tracker_cfg.get('ttl', TRACK_LIFECYCLE_TTL),
    save_dir=_tracker_cfg.get('save_dir', 'track_logs'),
    snapshot_interval=_tracker_cfg.get('snapshot_interval', TRACK_LIFECYCLE_SNAP_INTERVAL),
    frame_window_span=_tracker_cfg.get('frame_window_span', TRACK_LIFECYCLE_WINDOWS_SPAN),
)
