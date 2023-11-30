from scipy.spatial.transform.rotation import Rotation
import copy
import numpy as np
import numpy.typing as npt


class PositionVector:
    """A simple numpy based Vector class for positions"""

    def __init__(self, *args):
        if len(args) == 3:
            self._data: npt.NDArray[np.float64] = np.array(args)
        elif (
            len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].shape == (3,)
        ):
            self._data: npt.NDArray[np.float64] = args[0].copy()
        else:
            self._data: npt.NDArray[np.float64] = np.zeros(shape=(3,))

    def __eq__(self, other) -> bool:
        return np.array_equal(self._data, other._data)

    def __str__(self) -> str:
        return f"PositionVector({self._data[0]}, {self._data[1]}, {self._data[2]})"

    def __setitem__(self, i: int, value: float):
        if i < 3 and i >= 0:
            self._data[i] = value
        else:
            raise ValueError("Out of bounds.")

    def __getitem__(self, i: int) -> float:
        return float(self._data[i])

    def __iter__(self):
        return self._data.__iter__()

    def __add__(self, other):
        return PositionVector(self._data + other._data)

    def __sub__(self, other):
        return PositionVector(self._data - other._data)

    @property
    def x(self) -> float:
        return self._data[0]

    @x.setter
    def x(self, x: float):
        self._data[0] = x

    @property
    def y(self) -> float:
        return self._data[1]

    @y.setter
    def y(self, y: float):
        self._data[1] = y

    @property
    def z(self) -> float:
        return self._data[2]

    @z.setter
    def z(self, z: float):
        self._data[2] = z

    def normalize(self) -> 'PositionVector':
        norm = self.length()
        if norm != 0:
            self._data /= norm
        return self

    def length(self) -> float:
        return np.linalg.norm(self._data)

    def as_array(self) -> npt.NDArray[np.float64]:
        return self._data


class WaypointVolume:
    def __init__(self, bead_width: float, bead_height: float, extrusion_scale: float):
        """
        extrusion_scale: the relative amount of material to be extruded
        bead_width: bead width
        bead_height: bead height
        """
        self.bead_width = bead_width
        self.bead_height = bead_height
        self.extrusion_scale = extrusion_scale

    def __eq__(self, other):
        return (
            self.bead_height == other.bead_height
            and self.bead_width == other.bead_width
            and self.extrusion_scale == other.extrusion_scale
        )

    def serialize(self):
        return self.bead_width, self.bead_height, self.extrusion_scale

    def __str__(self):
        return f"WaypointVolume(bead_width={self.bead_width}, bead_height={self.bead_height}, extrusion_scale={self.extrusion_scale})"


class Waypoint:
    def __init__(self, *args):
        super().__init__()
        self._volume : WaypointVolume = WaypointVolume(0.0, 0.0, 0.0)
        self._position : PositionVector = PositionVector(0.0, 0.0, 0.0)
        self._rotation : Rotation = Rotation([0.0, 0.0, 0.0, 1.0])

        num_args = len(args)

        if num_args >= 1:
            if isinstance(args[0], np.ndarray) and args[0].shape == (3,):
                self._position = PositionVector(args[0])
            else:
                raise TypeError('First argument must be of type "Vector".')

        if num_args >= 2:
            if isinstance(args[1], Rotation):
                self._rotation = args[1]
            elif isinstance(args[1], np.ndarray) and args[1].shape == (3,):
                self.normal = args[1]
            else:
                raise TypeError('Second arg must be of type "Rotation" or "Vector".')

        if num_args == 3:
            if isinstance(args[2], WaypointVolume):
                self._volume = args[2]
            else:
                raise TypeError('Third argument must by of type "WaypointVolume".')

        if num_args > 3:
            raise Exception("Too many arguments")

    def __str__(self):
        return "Waypoint(X={}, Y={}, Z={}, x0={}, x1i={}, x2j={}, x3k={}, volume={})".format(
            *self._position, *self._rotation.as_quat(), self._volume
        )

    def __eq__(self, other):
        return (
            self._volume == other.volume
            and np.allclose(self._position, other._position)
            and np.allclose(self._rotation.as_quat(), other._rotation.as_quat())
        )

    @property
    def volume(self) -> WaypointVolume:
        return self._volume

    @volume.setter
    def volume(self, volume: WaypointVolume):
        if not isinstance(volume, WaypointVolume):
            raise TypeError("This only supports WaypointVolume.")
        self._volume = volume

    @property
    def position(self) -> PositionVector:
        """For legacy reasons, this returns not the Vector, but the underlying numpy array

        Returns:
            numpy.ndarray: array containing the position data [x,y,z]
        """

        return self._position

    @position.setter
    def position(self, pos: PositionVector):
        """For legacy reasons, this manipulates not the vector, but the underlying numpy array directly

        Args:
            pos (np.ndarray): _description_
        """
        if not isinstance(pos, PositionVector):
            raise TypeError("This only supports PositionVector.")
        self._position = pos

    @property
    def x(self) -> float:
        return self._position[0]

    @x.setter
    def x(self, x: float):
        self._position[0] = x

    @property
    def y(self) -> float:
        return self._position[1]

    @y.setter
    def y(self, y: float):
        self._position[1] = y

    @property
    def z(self) -> float:
        return self._position[2]

    @z.setter
    def z(self, z: float):
        self._position[2] = z

    # NOTE: the order of euler angles depends on the machine, so this is only true for some cases
    # It's best not to use these angles directly, but to derive them from the quaternions in a machine specific post-processor
    @property
    def a(self) -> float:
        return self._rotation.as_euler("XYZ", degrees=True)[0]

    @a.setter
    def a(self, value):
        self._rotation = Rotation.from_euler(
            "XYZ", [value, self.b, self.c], degrees=True
        )

    @property
    def b(self) -> float:
        return self._rotation.as_euler("XYZ", degrees=True)[1]

    @b.setter
    def b(self, value: float):
        self._rotation = Rotation.from_euler(
            "XYZ", [self.a, value, self.c], degrees=True
        )

    @property
    def c(self) -> float:
        return self._rotation.as_euler("XYZ", degrees=True)[2]

    @c.setter
    def c(self, value: float):
        self._rotation = Rotation.from_euler(
            "XYZ", [self.a, self.b, value], degrees=True
        )

    @property
    def normal(self) -> npt.NDArray[np.float64]:
        return self._rotation.apply(np.array([0.0, 0.0, 1.0]))

    @normal.setter
    def normal(self, vec: npt.NDArray[np.float64]):
        # Port of FreeCAD's vector to vector rot implementation
        # see https://github.com/FreeCAD/FreeCAD/blob/master/src/Base/Rotation.cpp
        vec_len = np.linalg.norm(vec)
        if vec_len == 0:
            raise ValueError('Length of the Vector is "0.0".')

        vec_norm = vec / vec_len
        vecz_norm = np.array([0.0, 0.0, 1.0])
        dot = np.dot(vec_norm, vecz_norm)

        cross = np.cross(vecz_norm, vec_norm)
        cross_len = np.linalg.norm(cross)

        if cross_len == 0.0:
            # Parallel vectors
            if dot > 0:
                self._rotation = Rotation.from_quat((0.0, 0.0, 0.0, 1.0))
            else:
                self._rotation = Rotation.from_quat((0.0, 1.0, 0.0, 0.0))
        else:
            # Vectors not parallel
            cross_norm = cross / cross_len
            angle = np.arccos(dot)
            self._rotation = Rotation.from_rotvec(angle * cross_norm)

        # Finally a sanity assertion
        np.testing.assert_almost_equal(vec_norm, self.normal, 2)

    @property
    def rounded_normal(self) -> npt.NDArray[np.float64]:
        normal = self.normal.round(decimals=2)
        return normal / np.linalg.norm(normal)

    def copy(self):
        return copy.deepcopy(self)
