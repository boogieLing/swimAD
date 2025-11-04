# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '12.0.7'

from boxmot.boxmot.postprocessing.gsi import gsi
from boxmot.boxmot.tracker_zoo import create_tracker, get_tracker_config
from boxmot.boxmot.trackers.botsort.botsort import BotSort
from boxmot.boxmot.trackers.bytetrack.bytetrack import ByteTrack
from boxmot.boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.boxmot.trackers.hybridsort.hybridsort import HybridSort
from boxmot.boxmot.trackers.ocsort.ocsort import OcSort
from boxmot.boxmot.trackers.strongsort.strongsort import StrongSort
from boxmot.boxmot.trackers.imprassoc.imprassoctrack import ImprAssocTrack
from boxmot.boxmot.trackers.boosttrack.boosttrack import BoostTrack
from boxmot.boxmot.multiview_tool.grid_determine import GridDeterminer


TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'hybridsort', 'imprassoc', 'boosttrack']

TOOLS = ['GridDeterminer']

__all__ = ("__version__",
           "StrongSort", "OcSort", "ByteTrack", "BotSort", "DeepOcSort", "HybridSort", "ImprAssocTrack", "BoostTrack",
           "create_tracker", "get_tracker_config", "gsi", "GridDeterminer")
