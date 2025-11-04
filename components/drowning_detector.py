import os
import math
import logging
from enum import IntEnum
from dataclasses import dataclass
from itertools import groupby
from typing import List, Dict, Optional, Callable

import numpy as np
import pandas as pd
import yaml

from components.logger import setup_logger
from components.track_window_manager import TrackObservation
from constant import CONFIG_BASE_PATH


# ====== 类别 ID（例如水上/水下）======
class ClassID(IntEnum):
    HEAD_ABOVE = 1  # 水上头部（cls_id = 1）
    HEAD_BELOW = 2  # 水下头部（cls_id = 2）


# ====== 规则编号 ======
class RuleID(IntEnum):
    HEAD_UNDERWATER_RATIO = 1
    LOW_VELOCITY_AND_ACC = 2
    JITTER_ANGLE = 3
    LONG_UNDERWATER_STREAK = 4
    DROP_EVENT = 5
    LOW_POSITION_VARIANCE = 6
    REPEATED_LOCATIONS = 7
    FREQUENT_DIR_CHANGE = 8
    PATH_RATIO = 9


# ====== 规则描述映射 ======
RULE_DESC_MAP = {
    RuleID.HEAD_UNDERWATER_RATIO: "头部在水下比例过高",
    RuleID.LOW_VELOCITY_AND_ACC: "速度和加速度均过低",
    RuleID.JITTER_ANGLE: "抖动角度 > 0.4（不规则运动）",
    RuleID.LONG_UNDERWATER_STREAK: "水下轨迹 > 1秒",
    RuleID.DROP_EVENT: "掉落事件伴随突然的水下持续",
    RuleID.LOW_POSITION_VARIANCE: "位置方差低（固定不动）",
    RuleID.REPEATED_LOCATIONS: "重复访问同一位置 >= 3次",
    RuleID.FREQUENT_DIR_CHANGE: "频繁改变方向（角度 > 1 弧度）",
    RuleID.PATH_RATIO: "路径/位移比 > 2.5（之字形运动）"
}


# ====== 每条规则的结构化结果 ======
@dataclass
class RuleResult:
    rule_id: RuleID  # 枚举标识
    rule_name: str  # 人类可读描述
    value: object  # 规则值（可为 float、tuple）
    triggered: bool  # 是否触发


# ====== 溺水规则检测器核心类 ======
class DrowningDetector:
    """
    基于规则的溺水检测器，适用于滑窗轨迹检测系统（与 TrackWindowManager 联动）

    支持特性：
    - 自动分析轨迹轨道是否存在异常
    - 每条规则编号、描述、结构化结果
    - 日志可调试
    """

    def __init__(
            self,
            fps: Optional[int] = None,
            use_original_bbox: Optional[bool] = None,
            config: Optional[dict] = None,
            config_path: Optional[str] = None,
    ):
        """
        :param fps: 帧率（用于速度/时间计算），若为 None 则由配置文件提供
        :param use_original_bbox: 是否使用原始图像坐标（True）或 projected bbox（False）；
                                  若为 None 则由配置文件提供
        :param config: 直接传入配置（优先级高于文件）
        :param config_path: 配置文件路径（默认 `config/drowning.yaml`）
        """
        self.logger = setup_logger(self.__class__.__name__)  # type: logging.Logger
        self.df: Optional[pd.DataFrame] = None

        # 加载配置（dict > file > 内置默认）
        self.cfg = self._load_config(config=config, config_path=config_path)

        # 实例属性从配置初始化，显式入参优先生效
        self.fps: float = float(fps if fps is not None else self.cfg.get('fps', 20))
        cfg_uob = self.cfg.get('use_original_bbox', False)
        self.use_original_bbox: bool = bool(cfg_uob if use_original_bbox is None else use_original_bbox)

        # 规则阈值便捷引用
        thresholds = self.cfg.get('thresholds', {})
        self._t_head_ratio = float(thresholds.get('head_underwater_ratio', 0.7))
        vel = thresholds.get('velocity', {})
        self._t_mean_v = float(vel.get('mean_v_lt', 0.2))
        self._t_mean_a = float(vel.get('mean_a_lt', 0.1))
        self._t_jitter = float(thresholds.get('jitter_angle_gt', 0.4))
        self._t_underwater_streak = float(thresholds.get('underwater_streak_s', 3.0))
        drop = thresholds.get('drop_event', {})
        self._t_drop_first = float(drop.get('first_half_underwater_ratio_lt', 0.2))
        self._t_drop_second = float(drop.get('second_half_underwater_ratio_gte', 0.9))
        self._t_pos_var = float(thresholds.get('position_variance_le', 0.1))
        rep = thresholds.get('repeat_location', {})
        self._t_grid_size = float(rep.get('grid_size', 0.1))
        self._t_min_visits_per_cell = int(rep.get('min_visits_per_cell', 2))
        self._t_min_repeated_cells = int(rep.get('min_repeated_cells', 3))
        self._t_dir_change = float(thresholds.get('direction_change_angle_gt', 1.0))
        self._t_path_ratio = float(thresholds.get('path_ratio_gt', 2.5))

        eps = self.cfg.get('eps', {})
        self._eps_zero = float(eps.get('zero_disp', 1e-5))

        # 规则聚合门限
        self._min_observations: int = int(self.cfg.get('min_observations', 5))
        self._min_duration_s: float = float(self.cfg.get('min_duration_s', 2.0))
        self._required_rule_count: int = int(self.cfg.get('required_rule_count', 4))
        self._allow_drop_single: bool = bool(self.cfg.get('enable_drop_event_single', True))

    @staticmethod
    def _fmt(value) -> str:
        """格式化规则值用于摘要日志。"""
        if isinstance(value, (tuple, list)):
            return "(" + ", ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in value) + ")"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _estimate_fps(self, observations: List[TrackObservation]) -> float:
        if len(observations) < 2:
            return self.fps  # 回退到默认
        duration = observations[-1].timestamp - observations[0].timestamp
        return (len(observations) - 1) / duration if duration > 1e-3 else self.fps

    def as_hook_fn(self) -> Callable[[str, List[TrackObservation]], Optional[Dict]]:
        """返回适配 TrackWindowManager.detect_abnormal_tracks 的 hook 函数。

        说明：
        - 输入为 (track_id, observations)
        - 内部调用 `evaluate` 完成规则评估
        - 若存在异常，evaluate 已输出 WARNING 摘要；此处仅在 DEBUG 下补充一行
        """
        def hook_fn(track_id: str, observations: List[TrackObservation]) -> Optional[Dict]:
            result = self.evaluate(track_id, observations)
            # 评估函数已输出摘要，这里仅在调试时补充
            if result is not None:
                rule_names = [r['rule_name'] for r in result.get('triggered_rules', [])]
                self.logger.debug(f"[DrowningDetector] track_id={track_id} 触发规则(调试): {rule_names}")
            return result

        return hook_fn

    def evaluate(self, track_id: str, observations: List[TrackObservation]) -> Optional[Dict]:
        """
        核心评估逻辑（阈值可配）：
        - 将观测数据（bbox 中心点与分类）转换为 DataFrame（_convert_to_dataframe）
        - 依次执行 9 条规则（rule_1 ~ rule_9），得到结构化 RuleResult 列表
        - 触发条件：满足 `required_rule_count`，或（启用时）`DropEvent` 单独触发
        - 返回格式：{"alarm": True, "triggered_rules": [RuleResult...]}
        """
        if len(observations) < max(self._min_observations, self.fps):
            self.logger.info(
                f"[Skip] track_id={track_id} 帧数不足（{len(observations)} < {int(max(self._min_observations, self.fps))}），跳过评估")
            return None

        duration = observations[-1].timestamp - observations[0].timestamp
        if duration < self._min_duration_s:
            self.logger.info(f"[Skip] track_id={track_id} 轨迹时长 {duration:.2f}s < {self._min_duration_s:.1f}s，跳过评估")
            return None
        old_fps = self.fps
        est_fps = self._estimate_fps(observations)
        self.logger.debug(
            f"[FPS] track_id={track_id} duration={duration:.2f}s 动态FPS={est_fps:.2f} (原={self.fps})")
        self.fps = est_fps

        self._convert_to_dataframe(observations)
        if self.df is None:
            self.logger.warning("数据转换失败。")
            return None

        # 执行所有规则（阈值由 YAML 配置或默认值提供）
        results = [
            self.rule_1_head_ratio(track_id),
            self.rule_2_velocity_acc(track_id),
            self.rule_3_jitter(track_id),
            self.rule_4_underwater_streak(track_id),
            self.rule_5_drop_event(track_id),
            self.rule_6_low_variance(track_id),
            self.rule_7_repeat_location(track_id),
            self.rule_8_direction_change(track_id),
            self.rule_9_path_ratio(track_id),
        ]

        triggered = [r for r in results if r.triggered]

        # 判断逻辑：规则数量阈值 或 允许掉落事件单独触发
        alarm = (
            len(triggered) >= self._required_rule_count or
            (self._allow_drop_single and any(r.rule_id == RuleID.DROP_EVENT for r in triggered))
        )
        self.fps = old_fps
        if alarm:
            # 摘要日志：仅在触发时以 WARNING 输出关键信息（规则名与格式化数值）
            summary = ", ".join(
                f"[{int(r.rule_id)}]{r.rule_name}={self._fmt(r.value)}" for r in triggered
            )
            self.logger.warning(f"[ALARM] track_id={track_id} 触发{len(triggered)}条规则: {summary}")
            return {
                "alarm": True,
                "triggered_rules": [r.__dict__ for r in triggered]
            }
        else:
            # 未触发仅输出调试摘要
            self.logger.debug(f"[OK] track_id={track_id} 未触发（触发数={len(triggered)}）")
            return None

    def _convert_to_dataframe(self, observations: List[TrackObservation]):
        """
        将滑窗中的轨迹观测转换为 pandas.DataFrame 结构，用于规则计算
        """
        data = []
        for obs in observations:
            entry = obs.entry
            x1, y1, x2, y2 = entry.original_bbox if self.use_original_bbox else (entry.x1, entry.y1, entry.x2, entry.y2)
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            head_above = 1 if entry.cls_id == ClassID.HEAD_ABOVE else 0
            data.append({"x": x, "y": y, "head_above": head_above})

        self.df = pd.DataFrame(data) if data else None

    # ====== 规则定义 ======

    def rule_1_head_ratio(self, track_id: str) -> RuleResult:
        ratio = (self.df['head_above'] == 0).mean()
        self.logger.debug(f"[{track_id}] [Rule 1] 头部在水下比例= {ratio:.3f}")
        return RuleResult(
            RuleID.HEAD_UNDERWATER_RATIO,
            RULE_DESC_MAP[RuleID.HEAD_UNDERWATER_RATIO],
            ratio,
            ratio >= self._t_head_ratio,
        )

    def rule_2_velocity_acc(self, track_id: str) -> RuleResult:
        pos = self.df[['x', 'y']].values
        v = np.diff(pos, axis=0) * self.fps
        v_mag = np.linalg.norm(v, axis=1)
        a = np.diff(v_mag) * self.fps
        mean_v, mean_a = v_mag.mean(), np.mean(np.abs(a))
        self.logger.debug(f"[{track_id}] [Rule 2] 速度 = {mean_v:.3f}, 加速度 = {mean_a:.3f}")
        return RuleResult(
            RuleID.LOW_VELOCITY_AND_ACC,
            RULE_DESC_MAP[RuleID.LOW_VELOCITY_AND_ACC],
            (mean_v, mean_a),
            mean_v < self._t_mean_v and mean_a < self._t_mean_a,
        )

    def rule_3_jitter(self, track_id: str) -> RuleResult:
        pos = self.df[['x', 'y']].values
        angles = [math.atan2(pos[i][1] - pos[i - 1][1], pos[i][0] - pos[i - 1][0]) for i in range(1, len(pos))]
        jitter = np.mean([abs(angles[i] - angles[i - 1]) for i in range(1, len(angles))]) if len(angles) >= 2 else 0
        self.logger.debug(f"[{track_id}] [Rule 3] 抖动 = {jitter:.3f}")
        return RuleResult(
            RuleID.JITTER_ANGLE,
            RULE_DESC_MAP[RuleID.JITTER_ANGLE],
            jitter,
            jitter > self._t_jitter,
        )

    def rule_4_underwater_streak(self, track_id: str) -> RuleResult:
        head = (self.df['head_above'] == 0).values
        max_len = curr = 0
        for h in head:
            curr = curr + 1 if h else 0
            max_len = max(max_len, curr)
        duration = max_len / self.fps
        self.logger.debug(f"[{track_id}] [Rule 4] 最大水下持续时间 = {duration:.2f}s")
        return RuleResult(
            RuleID.LONG_UNDERWATER_STREAK,
            RULE_DESC_MAP[RuleID.LONG_UNDERWATER_STREAK],
            duration,
            duration >= self._t_underwater_streak,
        )

    def rule_5_drop_event(self, track_id: str) -> RuleResult:
        h = self.df['head_above'].values
        half = len(h) // 2
        f, b = (h[:half] == 0).mean(), (h[half:] == 0).mean()
        max_len = max([sum(1 for _ in g) for k, g in groupby(h) if k == 0] or [0])
        duration = max_len / self.fps
        self.logger.debug(
            f"[{track_id}] [Rule 5] drop_event: 前半水下比例={f:.2f}, 后半水下比例={b:.2f}, 最大水下持续={duration:.2f}s")
        triggered = f < self._t_drop_first and b >= self._t_drop_second
        return RuleResult(RuleID.DROP_EVENT, RULE_DESC_MAP[RuleID.DROP_EVENT], (f, b, duration), triggered)

    def rule_6_low_variance(self, track_id: str) -> RuleResult:
        pos = self.df[['x', 'y']].values
        centroid = np.mean(pos, axis=0)
        drift = np.mean(np.linalg.norm(pos - centroid, axis=1) ** 2)
        self.logger.debug(f"[{track_id}] [Rule 6] 位置方差 = {drift:.4f}")
        return RuleResult(
            RuleID.LOW_POSITION_VARIANCE,
            RULE_DESC_MAP[RuleID.LOW_POSITION_VARIANCE],
            drift,
            drift <= self._t_pos_var,
        )

    def rule_7_repeat_location(self, track_id: str) -> RuleResult:
        pos = self.df[['x', 'y']].values
        visited = {}
        for x, y in pos:
            key = (int(x // self._t_grid_size), int(y // self._t_grid_size))
            visited[key] = visited.get(key, 0) + 1
        repeat = sum(1 for v in visited.values() if v >= self._t_min_visits_per_cell)
        self.logger.debug(f"[{track_id}] [Rule 7] 重复网格数量 = {repeat}")
        return RuleResult(
            RuleID.REPEATED_LOCATIONS,
            RULE_DESC_MAP[RuleID.REPEATED_LOCATIONS],
            repeat,
            repeat >= self._t_min_repeated_cells,
        )

    def rule_8_direction_change(self, track_id: str) -> RuleResult:
        pos = self.df[['x', 'y']].values
        delta = pos[-1] - pos[0]
        if np.linalg.norm(delta) < self._eps_zero:
            return RuleResult(RuleID.FREQUENT_DIR_CHANGE, RULE_DESC_MAP[RuleID.FREQUENT_DIR_CHANGE], np.pi, True)
        angles = []
        for i in range(1, len(pos)):
            vi = pos[i] - pos[i - 1]
            if np.linalg.norm(vi) < self._eps_zero:
                continue
            cos_theta = np.dot(vi, delta) / (np.linalg.norm(vi) * np.linalg.norm(delta))
            angles.append(np.arccos(np.clip(cos_theta, -1, 1)))
        avg_angle = np.mean(angles) if angles else 0
        self.logger.debug(f"[{track_id}] [Rule 8] 平均方向变化 = {avg_angle:.3f}")
        return RuleResult(
            RuleID.FREQUENT_DIR_CHANGE,
            RULE_DESC_MAP[RuleID.FREQUENT_DIR_CHANGE],
            avg_angle,
            avg_angle > self._t_dir_change,
        )

    def rule_9_path_ratio(self, track_id: str) -> RuleResult:
        pos = self.df[['x', 'y']].values
        path_len = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
        disp = np.linalg.norm(pos[-1] - pos[0])
        ratio = path_len / disp if disp > self._eps_zero else float('inf')
        self.logger.debug(f"[{track_id}] [Rule 9] 路径曲率比值 = {ratio:.3f}")
        return RuleResult(
            RuleID.PATH_RATIO,
            RULE_DESC_MAP[RuleID.PATH_RATIO],
            ratio,
            ratio > self._t_path_ratio,
        )

    # ====== 配置加载 ======
    @staticmethod
    def _default_config() -> dict:
        return {
            'use_original_bbox': False,
            'fps': 20,
            'min_observations': 5,
            'min_duration_s': 2.0,
            'required_rule_count': 4,
            'enable_drop_event_single': True,
            'thresholds': {
                'head_underwater_ratio': 0.7,
                'velocity': {
                    'mean_v_lt': 0.2,
                    'mean_a_lt': 0.1,
                },
                'jitter_angle_gt': 0.4,
                'underwater_streak_s': 3.0,
                'drop_event': {
                    'first_half_underwater_ratio_lt': 0.2,
                    'second_half_underwater_ratio_gte': 0.9,
                },
                'position_variance_le': 0.1,
                'repeat_location': {
                    'grid_size': 0.1,
                    'min_visits_per_cell': 2,
                    'min_repeated_cells': 3,
                },
                'direction_change_angle_gt': 1.0,
                'path_ratio_gt': 2.5,
            },
            'eps': {
                'zero_disp': 1e-5,
            },
        }

    def _load_config(self, config: Optional[dict], config_path: Optional[str]) -> dict:
        if config is not None:
            # 直接使用传入配置（允许上层完全控制）
            return config

        # 默认文件路径：config/drowning.yaml
        cfg_path = config_path or os.path.join(CONFIG_BASE_PATH, 'drowning.yaml')
        if not os.path.isfile(cfg_path):
            # 文件缺失，返回默认
            return self._default_config()
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                # 允许最外层为 {drowning: {...}} 或直接为 {...}
                drowning_cfg = data.get('drowning', data)
                # 用默认填充缺失项
                default = self._default_config()
                merged = self._deep_merge(default, drowning_cfg)
                return merged
        except Exception as e:
            # 解析失败，回退默认并记录一条可见日志
            logger = setup_logger(self.__class__.__name__)
            logger.warning(f"加载配置失败，使用默认配置。error={e}")
            return self._default_config()

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = DrowningDetector._deep_merge(result[k], v)
            else:
                result[k] = v
        return result
