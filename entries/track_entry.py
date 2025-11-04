import json
import time
from typing import Optional, List, Tuple


class DetectionEntry:
    def __init__(
            self,
            box_data: List[float],
            original_bbox: Tuple[float, float, float, float],
            view_name: str,
            track_id: Optional[str] = None,
    ):
        """
        box_data: [x1, y1, x2, y2, conf, cls_id] from projected box
        original_bbox: (x1, y1, x2, y2) in source image
        view_name: name of the camera view, e.g. "1", "2", etc.
        """
        self.x1, self.y1, self.x2, self.y2, self.conf, self.cls_id = box_data
        self.original_bbox = original_bbox
        self.view_name = view_name

        self.track_id = track_id  # True ID assigned by tracker

    def key(self) -> str:
        return f"{int(self.x1)}_{int(self.y1)}_{int(self.x2)}_{int(self.y2)}"

    def bbox(self) -> list:
        return [self.x1, self.y1, self.x2, self.y2]

    def box_data(self) -> list:
        return [self.x1, self.y1, self.x2, self.y2, self.conf, self.cls_id]

    def as_dict(self) -> dict:
        return {
            "bbox": self.bbox(),
            "conf": self.conf,
            "view_name": self.view_name,
            "cls_id": self.cls_id,
            "track_id": self.track_id,
            "original_bbox": self.original_bbox
        }

    def __repr__(self):
        return (
            f"DetectionEntry(track_id={self.track_id}, view_name={self.view_name}, "
            f"cls_id={self.cls_id}, "
            f"bbox={self.bbox()}, "
            f"original_bbox={self.original_bbox})"
        )


class TrackObservation:
    """
    单帧观测数据，封装了 DetectionEntry + 时间戳 + 帧号
    """

    def __init__(self, entry: DetectionEntry, frame_index: Optional[int] = None):
        self.entry = entry
        self.timestamp = time.time()
        self.frame_index = frame_index

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            **self.entry.as_dict()
        }

class AssociateResult:

    def __init__(self, bbox: list, conf: float, cls_id: float, track_id: str):
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.conf = conf
        self.cls_id = cls_id
        self.track_id = track_id

    def to_json(self) -> str:
        return json.dumps({
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "conf": self.conf,
            "cls_id": self.cls_id,
            "track_id": self.track_id,
        })

    def to_dict(self):
        return {
            "x1": self.x1, "y1": self.y1, "x2": self.x2,
            "y2": self.y2, "conf": self.conf, "cls_id": self.cls_id,
            "track_id": self.track_id
        }
