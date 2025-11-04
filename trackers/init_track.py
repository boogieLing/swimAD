from pathlib import Path

import yaml
from typing import Optional, Dict, Any, Type, List

from ultralytics.trackers.basetrack import BaseTrack

from constant import TRACKERS, TRACKER_CONFIGS

# tracker_type -> 类路径映射
TRACKER_MAPPING = {
    # TODO：更新映射
    'strongsort': 'boxmot.boxmot.trackers.strongsort.strongsort.StrongSort',
    'boosttrack': 'boxmot.boxmot.trackers.boosttrack.boosttrack.BoostTrack',
    'ocsort': 'boxmot.boxmot.trackers.ocsort.ocsort.OcSort',
}


def create_tracker(
        tracker_type: str,
        tracker_config: Optional[str] = None,
        reid_weights: Optional[str] = None,
        device: Optional[str] = None,
        half: Optional[bool] = None,
        per_class: Optional[bool] = None,
        evolve_param_dict: Optional[Dict[str, Any]] = None,
) -> BaseTrack:
    """
    创建跟踪器实例

    参数:
    - tracker_type: 跟踪器类型，如 'strongsort'
    - tracker_config: 跟踪器配置文件路径（若不使用 evolve_param_dict）
    - reid_weights: ReID 模型权重路径
    - device: 执行设备，如 'cuda' 或 'cpu'
    - half: 是否使用半精度
    - per_class: 是否按类别进行跟踪（部分模型不支持）
    - evolve_param_dict: 可选参数配置，若提供则不再读取 tracker_config

    返回:
    - 对应类型的跟踪器实例
    """

    # 从配置文件加载参数（或使用传入字典）
    if evolve_param_dict is None:
        if not tracker_config:
            raise ValueError("未提供 evolve_param_dict 时，tracker_config 必须提供")
        with open(tracker_config, "r") as f:
            yaml_config: Dict[str, Dict[str, Any]] = yaml.load(f, Loader=yaml.FullLoader)
            tracker_args = {k: v['default'] for k, v in yaml_config.items()}
    else:
        tracker_args = evolve_param_dict

    # ReID 参数（可选）
    reid_args = {
        'reid_weights': reid_weights,
        'device': device,
        'half': half,
    }

    # 校验 tracker_type
    if tracker_type not in TRACKER_MAPPING:
        raise ValueError(f"未知的 tracker_type: {tracker_type}")

    # 导入类
    module_path, class_name = TRACKER_MAPPING[tracker_type].rsplit('.', 1)
    tracker_class: Type = getattr(__import__(module_path, fromlist=[class_name]), class_name)

    # 处理特定 tracker 的参数组合
    if tracker_type in ['strongsort', 'boosttrack']:
        tracker_args.update(reid_args)
        tracker_args.pop('per_class', None)  # strongsort 不支持 per_class
    else:
        tracker_args['per_class'] = per_class

    return tracker_class(**tracker_args)


def init_trackers(
        tracking_method: str,
        batch_size: int = 1,  # 默认值 = 1
        reid_model: Optional[str] = None,
        device: Optional[str] = 'cpu',
        half: Optional[bool] = False,
        per_class: Optional[bool] = False,
) -> List[BaseTrack]:
    """
    批量初始化 trackers 列表（用于多视角或多batch处理）

    参数:
    - tracking_method: 跟踪器方法名称
    - batch_size: 初始化多少个 tracker 实例
    - reid_model: ReID 模型路径
    - device: 执行设备
    - half: 是否使用半精度
    - per_class: 是否按类别追踪

    返回:
    - Tracker 实例列表
    """

    assert tracking_method in TRACKERS, \
        f"未支持的 tracking_method: {tracking_method}，支持类型：{TRACKERS}"

    config_path = f"{TRACKER_CONFIGS}/{tracking_method}.yaml"
    if reid_model is not None:
        reid_model = Path(reid_model)
    trackers = []
    for _ in range(batch_size):
        tracker = create_tracker(
            tracker_type=tracking_method,
            tracker_config=str(config_path),
            reid_weights=reid_model,
            device=device,
            half=half,
            per_class=per_class,
        )
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    return trackers
