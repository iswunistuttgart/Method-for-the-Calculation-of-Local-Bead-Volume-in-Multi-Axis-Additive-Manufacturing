from Waypoint import Waypoint
import numpy.typing as npt
import numpy as np
from copy import deepcopy


class PathSegment:
    start_pose: Waypoint
    end_pose: Waypoint

class PrintSegment(PathSegment):
    occupying_voxels_indices: npt.NDArray[np.int64]
    slicer_volume: float

    calculated_volume: float
    adjusted_volume: float
    calculated_width: float
    calculated_height: float
    num_shared_voxels: int = 0

    def copy(self):
        return deepcopy(self)

    @property
    def length(self) -> float:
        return float((self.end_pose.position - self.start_pose.position).length())


class RetractionSegment(PrintSegment):
    retraction_distance: float = 0
