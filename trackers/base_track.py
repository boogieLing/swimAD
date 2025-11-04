import colorsys
import hashlib
import logging
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any

import numpy as np
import cv2 as cv

from components.logger import setup_logger


def iou_obb_pair(i, j, bboxes1, bboxes2):
    """
    Compute IoU for the rotated rectangles at index i and j in the batches `bboxes1`, `bboxes2` .
    """
    rect1 = bboxes1[int(i)]
    rect2 = bboxes2[int(j)]

    (cx1, cy1, w1, h1, angle1) = rect1[0:5]
    (cx2, cy2, w2, h2, angle2) = rect2[0:5]

    r1 = ((cx1, cy1), (w1, h1), angle1)
    r2 = ((cx2, cy2), (w2, h2), angle2)

    # Compute intersection
    ret, intersect = cv.rotatedRectangleIntersection(r1, r2)
    if ret == 0 or intersect is None:
        return 0.0  # No intersection

    # Calculate intersection area
    intersection_area = cv.contourArea(intersect)

    # Calculate union area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    # Compute IoU
    return intersection_area / union_area if union_area > 0 else 0.0


class AssociationFunction:
    def __init__(self, w, h, asso_mode="iou"):
        """
        Initializes the AssociationFunction class with the necessary parameters for bounding box operations.
        The association function is selected based on the `asso_mode` string provided during class creation.

        Parameters:
        w (int): The width of the frame, used for normalizing centroid distance.
        h (int): The height of the frame, used for normalizing centroid distance.
        asso_mode (str): The association function to use (e.g., "iou", "giou", "centroid", etc.).
        """
        self.w = w
        self.h = h
        self.asso_mode = asso_mode
        self.asso_func = self._get_asso_func(asso_mode)

    @staticmethod
    def iou_batch(bboxes1, bboxes2) -> np.ndarray:
        bboxes2 = np.expand_dims(bboxes2, 0)
        bboxes1 = np.expand_dims(bboxes1, 1)

        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        o = wh / (
                (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
                (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) -
                wh
        )
        return o

    @staticmethod
    def iou_batch_obb(bboxes1, bboxes2) -> np.ndarray:
        n, m = len(bboxes1), len(bboxes2)

        def wrapper(i, j):
            return iou_obb_pair(i, j, bboxes1, bboxes2)

        iou_matrix = np.fromfunction(np.vectorize(wrapper), shape=(n, m), dtype=int)
        return iou_matrix

    @staticmethod
    def hmiou_batch(bboxes1, bboxes2):
        """
        Compute a modified Intersection over Union (hIoU) between two batches of bounding boxes,
        incorporating a vertical overlap ratio.

        Parameters:
        - bboxes1: (N, 4) array of bounding boxes [x1, y1, x2, y2]
        - bboxes2: (M, 4) array of bounding boxes [x1, y1, x2, y2]

        Returns:
        - hmiou: (N, M) array where hmiou[i, j] is the modified IoU between bboxes1[i] and bboxes2[j]
        """
        # Expand dimensions for broadcasting
        bboxes1 = np.expand_dims(bboxes1, axis=1)  # Shape: (N, 1, 4)
        bboxes2 = np.expand_dims(bboxes2, axis=0)  # Shape: (1, M, 4)

        # Compute vertical overlap ratio 'o'
        intersect_y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        intersect_y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        intersection_height = np.maximum(0.0, intersect_y2 - intersect_y1)

        union_y1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        union_y2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
        union_height = np.maximum(1e-10, union_y2 - union_y1)

        o = intersection_height / union_height

        # Compute standard IoU
        inter_x1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        inter_y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        inter_x2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        inter_y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])  # Shape: (N, 1)
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])  # Shape: (1, M)

        union_area = area1 + area2 - inter_area

        iou = inter_area / (union_area + 1e-10)

        # Modify IoU with vertical overlap ratio
        hmiou = iou * o

        return hmiou

    @staticmethod
    def giou_batch(bboxes1, bboxes2) -> np.ndarray:
        """
        :param bboxes1: predict of bbox(N,4)(x1,y1,x2,y2)
        :param bboxes2: groundtruth of bbox(N,4)(x1,y1,x2,y2)
        :return:
        """
        # Ensure predict's bbox form
        bboxes2 = np.expand_dims(bboxes2, 0)
        bboxes1 = np.expand_dims(bboxes1, 1)

        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h  # Intersection area

        # Compute areas of individual boxes
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

        # Union area
        union_area = area1 + area2 - wh

        iou = wh / union_area

        xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
        yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
        yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
        wc = xxc2 - xxc1
        hc = yyc2 - yyc1
        assert (wc > 0).all() and (hc > 0).all()
        area_enclose = wc * hc  # Area of the smallest enclosing box

        # Corrected GIoU computation
        giou = iou - (area_enclose - union_area) / area_enclose
        giou = (giou + 1.0) / 2.0  # Resize from (-1,1) to (0,1)
        return giou

    def centroid_batch(self, bboxes1, bboxes2) -> np.ndarray:
        centroids1 = np.stack(((bboxes1[..., 0] + bboxes1[..., 2]) / 2,
                               (bboxes1[..., 1] + bboxes1[..., 3]) / 2), axis=-1)
        centroids2 = np.stack(((bboxes2[..., 0] + bboxes2[..., 2]) / 2,
                               (bboxes2[..., 1] + bboxes2[..., 3]) / 2), axis=-1)

        centroids1 = np.expand_dims(centroids1, 1)
        centroids2 = np.expand_dims(centroids2, 0)

        distances = np.sqrt(np.sum((centroids1 - centroids2) ** 2, axis=-1))
        norm_factor = np.sqrt(self.w ** 2 + self.h ** 2)
        normalized_distances = distances / norm_factor

        return 1 - normalized_distances

    def centroid_batch_obb(self, bboxes1, bboxes2) -> np.ndarray:
        centroids1 = np.stack((bboxes1[..., 0], bboxes1[..., 1]), axis=-1)
        centroids2 = np.stack((bboxes2[..., 0], bboxes2[..., 1]), axis=-1)

        centroids1 = np.expand_dims(centroids1, 1)
        centroids2 = np.expand_dims(centroids2, 0)

        distances = np.sqrt(np.sum((centroids1 - centroids2) ** 2, axis=-1))
        norm_factor = np.sqrt(self.w ** 2 + self.h ** 2)
        normalized_distances = distances / norm_factor

        return 1 - normalized_distances

    @staticmethod
    def ciou_batch(in_bboxes1, in_bboxes2) -> np.ndarray:
        """
        Calculate Complete Intersection over Union (CIoU) for batches of bounding boxes.

        :param in_bboxes1: Predicted bounding boxes of shape (N, 4) as (x1, y1, x2, y2)
        :param in_bboxes2: Ground truth bounding boxes of shape (N, 4) as (x1, y1, x2, y2)
        :return: CIoU scores scaled between 0 and 1
        """
        epsilon = 1e-7  # Small value to prevent division by zero

        # Expand dimensions for broadcasting
        bboxes2 = np.expand_dims(in_bboxes2, 0)  # Shape: (1, M, 4)
        bboxes1 = np.expand_dims(in_bboxes1, 1)  # Shape: (N, 1, 4)

        # Calculate the intersection box
        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h

        # Calculate IoU
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        iou = wh / (area1 + area2 - wh + epsilon)

        # Calculate center points
        centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
        centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
        centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
        centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

        # Calculate squared center distance
        inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

        # Calculate smallest enclosing box diagonal
        xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
        yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
        yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
        outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2 + epsilon

        # Calculate aspect ratio consistency
        w1 = bboxes1[..., 2] - bboxes1[..., 0]
        h1 = bboxes1[..., 3] - bboxes1[..., 1]
        w2 = bboxes2[..., 2] - bboxes2[..., 0]
        h2 = bboxes2[..., 3] - bboxes2[..., 1]

        # Prevent division by zero
        h2 = h2 + epsilon
        h1 = h1 + epsilon
        arctan_diff = np.arctan(w2 / h2) - np.arctan(w1 / h1)
        v = (4 / (np.pi ** 2)) * (arctan_diff ** 2)

        # Calculate alpha
        S = 1 - iou
        alpha = v / (S + v + epsilon)

        # Compute CIoU
        ciou = iou - (inner_diag / outer_diag) + (alpha * v)

        # Scale CIoU to [0, 1]
        return (ciou + 1) / 2.0

    @staticmethod
    def diou_batch(in_bboxes1, in_bboxes2) -> np.ndarray:
        """
        :param in_bboxes1: predict of bbox(N,4)(x1,y1,x2,y2)
        :param in_bboxes2: groundtruth of bbox(N,4)(x1,y1,x2,y2)
        :return:
        """
        # for details should go to https://arxiv.org/pdf/1902.09630.pdf
        # ensure predict's bbox form
        bboxes2 = np.expand_dims(in_bboxes2, 0)
        bboxes1 = np.expand_dims(in_bboxes1, 1)

        # calculate the intersection box
        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        iou = wh / (
                (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
                (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) -
                wh
        )

        centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
        centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
        centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
        centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

        inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

        xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
        yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
        yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

        outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
        diou = iou - inner_diag / outer_diag

        return (diou + 1) / 2.0

    @staticmethod
    def run_asso_func(self, bboxes1, bboxes2):
        """
        Runs the selected association function (based on the initialization string) on the input bounding boxes.

        Parameters:
        bboxes1: First set of bounding boxes.
        bboxes2: Second set of bounding boxes.
        """
        return self.asso_func(bboxes1, bboxes2)

    def _get_asso_func(self, asso_mode):
        """
        Returns the corresponding association function based on the provided mode string.

        Parameters:
        asso_mode (str): The association function to use (e.g., "iou", "giou", "centroid", etc.).

        Returns:
        function: The appropriate function for the association calculation.
        """
        asso_func = {
            "iou": AssociationFunction.iou_batch,
            "iou_obb": AssociationFunction.iou_batch_obb,
            "hmiou": AssociationFunction.hmiou_batch,
            "giou": AssociationFunction.giou_batch,
            "ciou": AssociationFunction.ciou_batch,
            "diou": AssociationFunction.diou_batch,
            "centroid": self.centroid_batch,  # only not being staticmethod
            "centroid_obb": self.centroid_batch_obb
        }

        if self.asso_mode not in asso_func:
            raise ValueError(f"Invalid association mode: {self.asso_mode}. Choose from {list(asso_func.keys())}")

        return asso_func[self.asso_mode]


class TrackState(IntEnum):
    """
    表示轨迹状态的枚举类。
    """
    New = 0  # 新建
    Tracked = 1  # 正在跟踪
    Lost = 2  # 暂时丢失
    LongLost = 3  # 丢失较久
    Removed = 4  # 已移除


class BaseTracker(ABC):
    def __init__(
            self,
            det_thresh: float = 0.3,
            max_age: int = 30,
            min_hits: int = 3,
            iou_threshold: float = 0.3,
            max_obs: int = 50,
            nr_classes: int = 80,
            per_class: bool = False,
            asso_func: str = 'iou',
            is_obb: bool = False
    ):
        """
        Initialize the BaseTracker object with detection threshold, maximum age, minimum hits,
        and Intersection Over Union (IOU) threshold for tracking objects in video frames.

        Parameters:
        - det_thresh (float): Detection threshold for considering detections.
        - max_age (int): Maximum age of a track before it is considered lost.
        - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
        - iou_threshold (float): IOU threshold for determining match between detection and tracks.

        Attributes:
        - frame_count (int): Counter for the frames processed.
        - active_tracks (list): List to hold active tracks, may be used differently in subclasses.
        """
        self.logger = setup_logger(self.__class__.__name__)  # type: logging.Logger
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.per_class = per_class  # Track per class or not
        self.nr_classes = nr_classes
        self.iou_threshold = iou_threshold
        self.last_emb_size = None
        self.asso_func_name = asso_func + "_obb" if is_obb else asso_func
        self.is_obb = is_obb

        self.window_size = 10
        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes
        self.per_class_active_tracks = None
        self._first_frame_processed = False  # Flag to track if the first frame has been processed
        self._first_dets_processed = False

        self.cls_map = {
            '0': 'ashore',
            '1': 'above',
            '2': 'under'
        }
        # Initialize per-class active tracks
        if self.per_class:
            self.per_class_active_tracks = {}
            for i in range(self.nr_classes):
                self.per_class_active_tracks[i] = []

        if self.max_age >= self.max_obs:
            self.logger.warning("Max age > max observations, increasing size of max observations...")
            self.max_obs = self.max_age + 5
            print("self.max_obs", self.max_obs)

    @abstractmethod
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Abstract method to update the tracker with new detections for a new frame. This method
        should be implemented by subclasses.

        Parameters:
        - dets (np.ndarray): Array of detections for the current frame.
        - img (np.ndarray): The current frame as an image array.
        - embs (np.ndarray, optional): Embeddings associated with the detections, if any.

        Raises:
        - NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("The update method needs to be implemented by the subclass.")

    def get_class_dets_n_embs(self, dets, embs, cls_id):
        # Initialize empty arrays for detections and embeddings
        class_dets = np.empty((0, 6))
        class_embs = np.empty((0, self.last_emb_size)) if self.last_emb_size is not None else None

        # Check if there are detections
        if dets.size > 0:
            class_indices = np.where(dets[:, 5] == cls_id)[0]
            class_dets = dets[class_indices]

            if embs is not None:
                # Assert that if embeddings are provided, they have the same number of elements as detections
                assert dets.shape[0] == embs.shape[
                    0], "Detections and embeddings must have the same number of elements when both are provided"

                if embs.size > 0:
                    class_embs = embs[class_indices]
                    self.last_emb_size = class_embs.shape[1]  # Update the last known embedding size
                else:
                    class_embs = None
        return class_dets, class_embs

    @staticmethod
    def setup_decorator(method):
        """
        Decorator to perform setup on the first frame only.
        This ensures that initialization tasks (like setting the association function) only
        happen once, on the first frame, and are skipped on subsequent frames.
        """

        def wrapper(self, *args, **kwargs):
            # If setup hasn't been done yet, perform it
            # Even if dets is empty (e.g., shape (0, 7)), this check will still pass if it's Nx7
            if not self._first_dets_processed:
                dets = args[0]
                if dets is not None:
                    if dets.ndim == 2 and dets.shape[1] == 6:
                        self.is_obb = False
                        self._first_dets_processed = True
                    elif dets.ndim == 2 and dets.shape[1] == 7:
                        self.is_obb = True
                        self._first_dets_processed = True

            if not self._first_frame_processed:
                img = args[1]
                self.h, self.w = img.shape[0:2]
                self.asso_func = AssociationFunction(w=self.w, h=self.h, asso_mode=self.asso_func_name).asso_func

                # Mark that the first frame setup has been done
                self._first_frame_processed = True

            # Call the original method (e.g., update)
            return method(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def per_class_decorator(update_method):
        """
        Decorator for the update method to handle per-class processing.
        """

        def wrapper(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None):

            # handle different types of inputs
            if dets is None or len(dets) == 0:
                dets = np.empty((0, 6))

            if self.per_class:
                # Initialize an array to store the tracks for each class
                per_class_tracks = []

                # same frame count for all classes
                frame_count = self.frame_count

                for cls_id in range(self.nr_classes):
                    # Get detections and embeddings for the current class
                    class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)

                    self.logger.debug(
                        f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings {class_embs.shape if class_embs is not None else None}")

                    # Activate the specific active tracks for this class id
                    self.active_tracks = self.per_class_active_tracks[cls_id]

                    # Reset frame count for every class
                    self.frame_count = frame_count

                    # Update detections using the decorated method
                    tracks = update_method(self, dets=class_dets, img=img, embs=class_embs)

                    # Save the updated active tracks
                    self.per_class_active_tracks[cls_id] = self.active_tracks

                    if tracks.size > 0:
                        per_class_tracks.append(tracks)

                # Increase frame count by 1
                self.frame_count = frame_count + 1

                return np.vstack(per_class_tracks) if per_class_tracks else np.empty((0, 8))
            else:
                # Process all detections at once if per_class is False
                return update_method(self, dets=dets, img=img, embs=embs)

        return wrapper

    def check_inputs(self, dets, img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert (
                len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        if self.is_obb:
            assert (
                    dets.shape[1] == 7
            ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6 (cx,cy,w,h,angle,conf,cls)"
        else:
            assert (
                    dets.shape[1] == 6
            ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6 (x1,y1,x2,y2,conf,cls)"

    def id_to_color(self, id: int, saturation: float = 0.75, value: float = 0.95) -> tuple:
        """
        Generates a consistent unique BGR color for a given ID using hashing.

        Parameters:
        - id (int): Unique identifier for which to generate a color.
        - saturation (float): Saturation value for the color in HSV space.
        - value (float): Value (brightness) for the color in HSV space.

        Returns:
        - tuple: A tuple representing the BGR color.
        """

        # Hash the ID to get a consistent unique value
        hash_object = hashlib.sha256(str(id).encode())
        hash_digest = hash_object.hexdigest()

        # Convert the first few characters of the hash to an integer
        # and map it to a value between 0 and 1 for the hue
        hue = int(hash_digest[:8], 16) / 0xffffffff

        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)

        # Convert RGB from 0-1 range to 0-255 range and format as hexadecimal
        rgb_255 = tuple(int(component * 255) for component in rgb)
        hex_color = '#%02x%02x%02x' % rgb_255
        # Strip the '#' character and convert the string to RGB integers
        rgb = tuple(int(hex_color.strip('#')[i:i + 2], 16) for i in (0, 2, 4))

        # Convert RGB to BGR for OpenCV
        bgr = rgb[::-1]

        return bgr

    def plot_trackers_trajectories_origin(
            self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2,
            fontscale: float = 0.5,
    ) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """

        if self.is_obb:

            angle = box[4] * 180.0 / np.pi  # Convert radians to degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            # print((width, height))
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)  # Convert to integer

            # Draw the rectangle on the image
            img = cv.polylines(img, [box_poly], isClosed=True, color=self.id_to_color(id), thickness=thickness)

            img = cv.putText(
                img,
                f'{int(id)}, c: {int(cls)}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        else:
            img = cv.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.id_to_color(id),
                thickness
            )
            img = cv.putText(
                img,
                f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        return img

    def plot_plain_box_on_img(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2,
                              fontscale: float = 0.5) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """

        if self.is_obb:

            angle = box[4] * 180.0 / np.pi  # Convert radians to degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            # print((width, height))
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)  # Convert to integer

            # Draw the rectangle on the image
            img = cv.polylines(img, [box_poly], isClosed=True, color=self.id_to_color(id), thickness=thickness)

            img = cv.putText(
                img,
                f'{int(id)}, {self.cls_map[str(cls)]}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        else:
            img = cv.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.id_to_color(id),
                thickness
            )
            txt = f'{int(id)}, {self.cls_map[str(int(cls))]}'
            text_size, _ = cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, fontscale, thickness)
            cv.rectangle(
                img,
                (int(box[0]), int(box[1]) - 10 - text_size[1]),
                (int(box[0]) + text_size[0], int(box[1]) + 10),
                (255, 255, 255),
                -1
            )
            img = cv.putText(
                img,
                txt,
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        return img

    def plot_box_on_img(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2,
                        fontscale: float = 0.5) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """
        if self.is_obb:

            angle = box[4] * 180.0 / np.pi  # Convert radians to degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            # print((width, height))
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)  # Convert to integer

            # Draw the rectangle on the image
            img = cv.polylines(img, [box_poly], isClosed=True, color=self.id_to_color(id), thickness=thickness)

            img = cv.putText(
                img,
                f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        else:
            img = cv.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.id_to_color(id),
                thickness
            )
            img = cv.putText(
                img,
                f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        return img

    def plot_box_on_img_with_rule(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2,
                                  fontscale: float = 0.5, rule=None) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """
        # pdb.set_trace()
        if self.is_obb:

            angle = box[4] * 180.0 / np.pi  # Convert radians to degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            # print((width, height))
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)  # Convert to integer

            # Draw the rectangle on the image
            img = cv.polylines(img, [box_poly], isClosed=True, color=(0, 0, 255), thickness=thickness)

            img = cv.putText(
                img,
                f'{int(id)}, {self.cls_map[str(cls)]}, True',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (0, 0, 255),  # red for AD objects
                thickness
            )
        else:
            img = cv.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 0, 255),  # red for AD objects
                thickness
            )
            img = cv.putText(
                img,
                f'{int(id)}, {self.cls_map[str(int(cls))]}, True',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (0, 0, 255),
                thickness
            )
        return img

    def plot_trackers_trajectories(self, img: np.ndarray, observations: list, id: int) -> np.ndarray:
        """
        Draws the trajectories of tracked objects based on historical observations. Each point
        in the trajectory is represented by a circle, with the thickness increasing for more
        recent observations to visualize the path of movement.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories.
        - observations (list): A list of bounding box coordinates representing the historical
        observations of a tracked object. Each observation is in the format (x1, y1, x2, y2).
        - id (int): The unique identifier of the tracked object for color consistency in visualization.

        Returns:
        - np.ndarray: The image array with the trajectories drawn on it.
        """
        for i, box in enumerate(observations):
            trajectory_thickness = int(np.sqrt(float(i + 1)) * 1.2)
            if self.is_obb:
                img = cv.circle(
                    img,
                    (int(box[0]), int(box[1])),
                    2,
                    self.id_to_color(id),  # red for AD objects
                    thickness=trajectory_thickness
                )
            else:

                img = cv.circle(
                    img,
                    (int((box[0] + box[2]) / 2),
                     int((box[1] + box[3]) / 2)),
                    2,
                    self.id_to_color(id),  # red for AD objects
                    thickness=trajectory_thickness
                )
        return img

    def plot_plain_results(self, img: np.ndarray, show_trajectories: bool, thickness: int = 2,
                           fontscale: float = 0.5) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
        - show_trajectories (bool): Whether to show the trajectories.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """

        # if values in dict
        if self.per_class_active_tracks is not None:
            for k in self.per_class_active_tracks.keys():
                active_tracks = self.per_class_active_tracks[k]
                for a in active_tracks:
                    if int(a.cls) == 0: continue  # ignore the ashore person
                    if a.history_observations:
                        if len(a.history_observations) > 2:
                            box = a.history_observations[-1]
                            img = self.plot_plain_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                            if show_trajectories:
                                img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        else:
            for a in self.active_tracks:
                if int(a.cls) == 0: continue  # ignore the ashore person
                if a.history_observations:
                    if len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        img = self.plot_plain_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                        if show_trajectories:
                            img = self.plot_trackers_trajectories(img, a.history_observations, a.id)

        return img

    def plot_multi_view_results(self, img: np.ndarray, show_trajectories: bool, thickness: int = 2,
                                fontscale: float = 0.5) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
        - show_trajectories (bool): Whether to show the trajectories.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """

        # if values in dict
        if self.per_class_active_tracks is not None:
            for k in self.per_class_active_tracks.keys():
                active_tracks = self.per_class_active_tracks[k]
                for a in active_tracks:
                    if a.history_observations:
                        if len(a.history_observations) > 2:
                            box = a.history_observations[-1]
                            img = self.plot_plain_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                            if show_trajectories:
                                img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        else:
            for a in self.active_tracks:
                if a.history_observations:
                    if len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        img = self.plot_plain_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                        if show_trajectories:
                            img = self.plot_trackers_trajectories(img, a.history_observations, a.id)

        return img

    def plot_results(self, img: np.ndarray, show_trajectories: bool, thickness: int = 2,
                     fontscale: float = 0.5) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
        - show_trajectories (bool): Whether to show the trajectories.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """

        # if values in dict
        if self.per_class_active_tracks is not None:
            for k in self.per_class_active_tracks.keys():
                active_tracks = self.per_class_active_tracks[k]
                for a in active_tracks:
                    if a.history_observations:
                        if len(a.history_observations) > 2:
                            box = a.history_observations[-1]
                            img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                            if show_trajectories:
                                img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        else:
            for a in self.active_tracks:
                if a.history_observations:
                    if len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                        if show_trajectories:
                            img = self.plot_trackers_trajectories(img, a.history_observations, a.id)

        return img

    def detect_ad(self) -> tuple[list[list[str | Any]], list[dict[str, Any]]]:
        """
        detect the AD swimmer with rules, all the rules function is named as rule+num num~[1,99]
        """

        ad_list = []
        info_list = []
        # pdb.set_trace()
        for a in self.active_tracks:
            match_rules = []
            if a.history_observations:
                for i in range(1, 99):
                    rule_name = f"rule{i}"
                    if hasattr(self, rule_name):
                        func = getattr(self, rule_name)
                        flag, info = func(a.history_observations, a.id)
                        info_list.append({f"rule{i}": info})
                        if flag:
                            match_rules.append(rule_name)

                if len(match_rules) > 0:
                    rules = " ".join(match_rules)
                    ad_list.append([a, rules])

        # if len(AD_list) > 0:
        #     pdb.set_trace()
        return ad_list, info_list

    def plot_ad_results(
            self, img: np.ndarray, show_trajectories: bool, AD_list: list, thickness: int = 4,
            fontscale: float = 0.5,
    ) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
        - show_trajectories (bool): Whether to show the trajectories.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """
        for a in AD_list:
            rule = a[-1]
            a = a[0]
            if a.history_observations:
                if len(a.history_observations) > 2:
                    # pdb.set_trace()
                    box = a.history_observations[-1]
                    img = self.plot_box_on_img_with_rule(img, box, a.conf, a.cls, a.id, thickness, fontscale, rule=rule)
                    if show_trajectories:
                        img = self.plot_trackers_trajectories(img, a.history_observations, a.id)

        return img

    def detect_ad_v2(self, object_map, metrics=None) -> dict[str, dict[str, list[Any] | Any]]:
        """
        detect the AD swimmer with rules, all the rules function is named as rule+num num~[1,99]
        """

        if metrics is None:
            metrics = []
        results = dict()
        for a in self.active_tracks:
            match_rules = []
            if a.history_observations:
                box = a.history_observations[-1][:4]
                box_xyxy = f'{int(box[0]):d}_{int(box[1]):d}_{int(box[2]):d}_{int(box[3]):d}'
                assert box_xyxy in object_map
                result = {'view': object_map[box_xyxy][0],
                          'bbox_left_top': [object_map[box_xyxy][1][0], object_map[box_xyxy][1][1]],
                          'bbox_right_bottom': [object_map[box_xyxy][1][2], object_map[box_xyxy][1][3]]}
                for metric in metrics:
                    assert hasattr(self, metric), f"The {metric} is not implemented."
                    func = getattr(self, metric)
                    value = func(a.history_observations, a.id)
                    result[f'{metric}'] = value
                results[f'obj{a.id}'] = result

        return results
