import copy
from Waypoint import Waypoint
from enum import Enum


class Coordinates(Enum):
    Absolute = 90
    Relative = 91


class MovementType(Enum):
    Undefined = 1000
    PTP = 0
    Linear = 1
    CircCW = 2
    CircCCW = 3


class MachineState:
    def __init__(self):
        self.coordinates: Coordinates = Coordinates.Absolute
        self.movement: MovementType = MovementType.Linear
        self.pose: Waypoint = Waypoint()
        self.e: float = 0
        self.e_calc: float = 0
        self.g: int = 0
        self.feed: float = 0
        self.retracted: bool = False
        self.extruding: bool = False

    def __eq__(self, other) -> bool:
        return (
            self.coordinates == other.coordinates
            and self.movement == other.movement
            and self.extruding == other.extruding
            and self.retracted == other.retracted
            and self.pose == other.pose
            and self.e == other.e
            and self.e_calc == other.e_calc
            and self.feed == other.feed
        )

    def update_from_dict(self, dictionary: dict[str, float]):
        if "X" in dictionary:
            self.pose.x = dictionary["X"]
        if "Y" in dictionary:
            self.pose.y = dictionary["Y"]
        if "Z" in dictionary:
            self.pose.z = dictionary["Z"]
        if "A" in dictionary:
            self.pose.a = dictionary["A"]
        if "B" in dictionary:
            self.pose.b = dictionary["B"]
        if "C" in dictionary:
            self.pose.c = dictionary["C"]
        if "E" in dictionary:
            self.e = dictionary["E"]
        if "F" in dictionary:
            self.feed = dictionary["F"]

    def __str__(self) -> str:
        # return f"MachineState(X={self.x}, Y={self.y}, Z={self.z}, A={self.a}, B={self.b}, E={self.e})"
        return f"MachineState(G{self.coordinates} G{self.movement} X={self.pose.x}, Y={self.pose.y}, Z={self.pose.z}, A={self.pose.a}, B={self.pose.b}, E={self.e}, F={self.feed}, extruding={self.extruding})"

    def copy(self) -> "MachineState":
        return copy.deepcopy(self)
