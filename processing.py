from MachineState import MachineState, Coordinates, MovementType
from PathSegment import PathSegment, PrintSegment, RetractionSegment
from Waypoint import PositionVector

from copy import deepcopy
from gcodeparser import GcodeParser, GcodeLine
from math import pi
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from typing import List, Tuple
from util import logtime
import numpy as np
from binvox_rw import Voxels


@logtime
def parse_gcode(parameters: dict, gcode: str) -> List[Tuple[GcodeLine, PathSegment]]:
    """This parses gcode only to a minimal extent to extract path start and end positions, orientation and volume.

    Args:
        parameters (dict): A dictionary that contains relevant parameters. Namely "filament_diameter" : float
        gcode (str): The "raw" gcode as a string.

    Returns:
        List[Tuple[GcodeParser.Gcode, PathSegment]]: Pairs of the original gcode line and the PathSegment objects extracted from it. This way, the original lines can only be modified where necessary, when the code is exported.
    """
    pairs = []
    last_state = MachineState()
    last_state.coordinates = Coordinates.Absolute
    # parse gcode and update state
    for line in GcodeParser(gcode, include_comments=True).lines:
        next_state = last_state.copy()
        path_segment: PathSegment = PathSegment()
        if line.command[0] == "G":
            match line.command[1]:
                case 0:  # Traversal Path
                    next_state.update_from_dict(line.params)
                    next_state.extruding = False
                    next_state.movement = MovementType.Linear
                case 1:  # This is either a print path or a traversal path
                    next_state.update_from_dict(line.params)
                    next_state.movement = MovementType.Linear
                    if "E" in line.params:
                        if next_state.e > last_state.e:
                            # print path
                            next_state.extruding = True
                            # Undo of retraction
                            if last_state.retracted:
                                next_state.retracted = False
                                next_state.e_calc += next_state.e - last_state.e
                                path_segment = RetractionSegment()
                                path_segment.retraction_distance = (
                                    next_state.e - last_state.e
                                )
                                line.comment += " undo retraction"
                            else:
                                # This is an actual print path
                                path_segment = PrintSegment()
                                path_segment.start_pose = last_state.pose.copy()
                                path_segment.end_pose = next_state.pose.copy()
                                path_segment.slicer_volume = (
                                    (next_state.e - last_state.e)
                                    * 0.25
                                    * parameters["filament_diameter"] ** 2
                                    * pi
                                )
                        elif next_state.e < last_state.e:
                            # Retraction
                            next_state.extruding = False
                            next_state.retracted = True
                            next_state.e_calc -= last_state.e - next_state.e
                            path_segment = RetractionSegment()
                            path_segment.retraction_distance = (
                                next_state.e - last_state.e
                            )
                            line.comment += " retraction"
                    else:
                        # Probably nothing relevant happening here
                        next_state.extruding = False
                case 92:
                    # This is used to set the extruder back to 0 so it doesn't overflow
                    next_state.update_from_dict(line.params)

            last_state = next_state
        pairs.append((line, path_segment))
    # return all lines with corresponding path segements.
    return pairs


@logtime
def interpolate_path_segments(
    parameters: dict, pairs: List[Tuple[GcodeLine, PathSegment]]
) -> List[Tuple[GcodeLine, PathSegment]]:
    """Inserts linearly interpolated points into the given pairs of gcode and segments.
    This allows for a higher resolution of the local path volume.

    Args:
        parameters (dict): The dictionary of parameters. "filament_diameter" and "interpolate_length" are used.
        pairs (List[Tuple[GcodeParser.Gcode, PathSegment]]): The input pairs of gcode and derived path segments.

    Returns:
        List[Tuple[GcodeParser.Gcode, PathSegment]]: The input pairs of gcode and derived path segments.
    """
    scaling_factor: float = 4 / (float(parameters["filament_diameter"]) ** 2 * pi)
    max_length: float = float(parameters["interpolate_length"])
    new_pairs : List[Tuple[GcodeLine, PathSegment]]= []
    for line, path_segment in pairs:
        # Don't split non-printing segments
        if not isinstance(path_segment, PrintSegment):
            new_pairs.append((line, path_segment))
            continue

        # split other segments
        if path_segment.length == 0:
            # Remove the zero-length segment
            continue

        num_segments = int(path_segment.length / max_length)
        if num_segments == 0:
            new_pairs.append((line, path_segment))
            continue

        direction_vec = (
            path_segment.end_pose.position.as_array()
            - path_segment.start_pose.position.as_array()
        ) / path_segment.length
        actual_segment_length = path_segment.length / num_segments

        new_segments = []
        new_lines = []
        for i in range(num_segments):
            seg = PrintSegment()
            li = deepcopy(line)
            seg.start_pose = (
                path_segment.start_pose.copy()
                if i == 0
                else new_segments[-1].end_pose.copy()
            )
            seg.end_pose = seg.start_pose.copy()
            seg.end_pose.position = PositionVector(
                seg.start_pose.position.as_array()
                + direction_vec * actual_segment_length
            )
            seg.slicer_volume = (
                path_segment.slicer_volume * actual_segment_length
            ) / path_segment.length
            new_segments.append(seg)

            li.update_param("X", round(float(seg.end_pose.position.x), 5))
            li.update_param("Y", round(float(seg.end_pose.position.y), 5))
            li.update_param("Z", round(float(seg.end_pose.position.z), 5))
            li.update_param("E", round(float(seg.slicer_volume * scaling_factor), 5))
            new_lines.append(li)

        new_pairs.extend(list(zip(new_lines, new_segments)))
    return new_pairs


def offset_path_segments(path_segments: List[PathSegment], bead_height: float) -> None:
    """Modifies the path segments by offsetting the TCP position which is usually given as the tip of the nozzle to the center of the bead.

    Args:
        path_segments (List[PathSegment]): List of path segments. Modifies the segments directly. As the relationship of the pairs should not be broken and we don't want to modify the original gcode lines.
        bead_height (float): Height of the bead or the layer.
    """
    for segment in path_segments:
        segment.start_pose.position = PositionVector(
            segment.start_pose.position.as_array()
            - segment.start_pose.normal * bead_height / 2
        )
        segment.end_pose.position = PositionVector(
            segment.end_pose.position.as_array()
            - segment.end_pose.normal * bead_height / 2
        )
    return


def parallel_determine_segment_voxels(
    parameters: dict, voxelgrid: Voxels, print_segments: List[PrintSegment]
):
    """Parallel processing of the give Print segments, using determine_segment_voxels and storing the results right away

    Args:
        parameters (dict): dict of the model parameters.
        voxelgrid (Voxels): grid of voxels as given by binvox_rw
        print_segments (List[PrintSegment]: print segments to iterate through. The indices of the voxels that are occupied by each segment are stored within PrintSegment.
    """

    # Somewhat hacky way to parallelize processingfrom binvox_rw import Voxels
    global determine_segment_voxels_helper

    def determine_segment_voxels_helper(segment):
        return determine_segment_voxels(parameters, voxelgrid, segment)

    with Pool() as pool:
        indices_list = pool.map(determine_segment_voxels_helper, print_segments)

    for segment, index in zip(print_segments, indices_list):
        segment.occupying_voxels_indices = index


@logtime
def trace_paths(
    parameters: dict, voxelgrid: Voxels, print_segments: List[PrintSegment]
) -> None:
    """Iterate through all printing segments of the paths and determines the voxels occupied by each path

    Args:
        parameters (dict): dict of the model parameters.
        voxelgrid (Voxels): grid of voxels as given by binvox_rw
        print_segments (List[PrintSegment]: print segments to iterate through. The indices of the voxels that are occupied by each segment are stored within PrintSegment.
    """
    parallel_determine_segment_voxels(parameters, voxelgrid, print_segments)
    for segment in print_segments:
        apply_occupancy(voxelgrid, segment)


def apply_occupancy(voxelgrid: Voxels, segment: PrintSegment) -> None:
    """Reads the indices of the given print segment and increases the bead counter for each voxel in the grid.

    Args:
        voxelgrid (Voxels): The global voxel grid.
        segment (PrintSegment): The print segment with identified voxels.
    """
    if not segment.occupying_voxels_indices[0].any():
        return

    i, j, k = segment.occupying_voxels_indices
    # Filter out voxels that are outside of the model
    mask = voxelgrid.data[i, j, k] > 0

    # Overflows here at 255 are very unlikely to occur and if it happens anyway, parameters are most likely very wrong.
    if np.any(voxelgrid.data[i[mask], j[mask], k[mask]] == 255):
        raise Exception("Overflow in grid cell. Check your parameters!")
    voxelgrid.data[i[mask], j[mask], k[mask]] += 1


def determine_segment_voxels(
    parameters: dict, voxelgrid: Voxels, segment: PrintSegment
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        parameters (dict): dict of the model parameters.
        voxelgrid (Voxels): grid of voxels as given by binvox_rw.
        segment (PrintSegment): segment to be processed.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Indices of the voxels occupied by this path segment. (x,y,z)
    """
    # get the defining path segment
    start_position = segment.start_pose.position.as_array()
    start_orientation = segment.start_pose.normal
    end_position = segment.end_pose.position.as_array()
    bead_width = parameters["bead"]["width"]["maximal"]
    bead_height = parameters["bead"]["height"]["maximal"]

    # Calculate direction vector
    direction_vec = end_position - start_position
    bead_length = np.linalg.norm(direction_vec)

    if bead_length == 0:
        return np.ndarray([]), np.ndarray([]), np.ndarray([])

    direction_vec = direction_vec / bead_length

    # ensure the orientation is normalized
    start_orientation = start_orientation / np.linalg.norm(start_orientation)

    # get third axis
    y_axis = np.cross(start_orientation, direction_vec)

    # calculate transformations
    # origin to segment
    rot_o_to_seg = Rotation.from_matrix(
        np.column_stack((direction_vec, y_axis, start_orientation))
    )

    # segment to origin (inverse)
    rot_seg_to_o = rot_o_to_seg.inv()
    trans_seg_to_o = -start_position

    # Calculate an axis aligned transformed bounding box for the segment shape, so the relevant voxels can be pre-selected
    bead_bounding_box_p1 = np.array(
        [
            min(start_position[0], end_position[0]) - bead_width / 2,
            min(start_position[1], end_position[1]) - bead_width / 2,
            min(start_position[2], end_position[2]) - bead_height / 2,
        ]
    )
    bead_bounding_box_p2 = np.array(
        [
            max(start_position[0], end_position[0]) + bead_width / 2,
            max(start_position[1], end_position[1]) + bead_width / 2,
            max(start_position[2], end_position[2]) + bead_height / 2,
        ]
    )
    # select all points within segment bounding box:
    idx_p1 = np.clip(
        voxelgrid.get_index_for_position(bead_bounding_box_p1),
        0,
        np.subtract(voxelgrid.dims, 1),
    )
    idx_p2 = np.clip(
        voxelgrid.get_index_for_position(bead_bounding_box_p2),
        0,
        np.subtract(voxelgrid.dims, 1),
    )

    # to include all voxels, increase idx_p2
    segment_indices = np.meshgrid(
        np.arange(idx_p1[0], idx_p2[0] + 1, dtype=int),
        np.arange(idx_p1[1], idx_p2[1] + 1, dtype=int),
        np.arange(idx_p1[2], idx_p2[2] + 1, dtype=int),
    )

    # transform voxels and check if they are in the segment definition
    voxels_x, voxels_y, voxels_z = voxelgrid.get_position_for_index(*segment_indices)

    # transform to shape (N, 3)
    voxels = np.array(
        [voxels_x.ravel(), voxels_y.ravel(), voxels_z.ravel()]
    ).transpose()

    # apply transformation
    transformed_voxels = rot_seg_to_o.apply(np.add(trans_seg_to_o, voxels))

    # restore original shape
    x = transformed_voxels[:, 0].reshape(voxels_x.shape)
    y = transformed_voxels[:, 1].reshape(voxels_y.shape)
    z = transformed_voxels[:, 2].reshape(voxels_z.shape)

    # identify paramters for the ellipsoid
    w = bead_width / 2
    h = bead_height / 2
    length = bead_length

    # mask voxels outside of the elongated ellipsoid
    ellipsoid1 = ((x**2 + y**2) / (w**2) + (z**2) / (h**2)) <= 1
    ellipsoid2 = (((x - length) ** 2 + y**2) / (w**2) + (z**2) / (h**2)) <= 1
    cylinder = (
        (((y**2) / (w**2) + (z**2) / (h**2)) <= 1) & (0 <= x) & (x <= length)
    )
    voxels_in_segment = ellipsoid1 | cylinder | ellipsoid2

    # Note: This can still contain voxels outside of the part, but that doesn't matter here

    return (
        segment_indices[0][voxels_in_segment],
        segment_indices[1][voxels_in_segment],
        segment_indices[2][voxels_in_segment],
    )


@logtime
def calc_volume(
    parameters: dict, voxelgrid: Voxels, print_segments: List[PrintSegment]
) -> None:
    """Calculates volume for all given PrintSegments in parallel. See calc_segment_volume for details.
    The results are constrained by get_adjusted_segment_volume

    Args:
        parameters (dict): dict of the model parameters.
        voxelgrid (Voxels): grid of voxels as given by binvox_rw.
        print_segments (List[PrintSegment]): List of segments to be processed.
    """
    global calc_segment_volume_helper

    def calc_segment_volume_helper(segment):
        return calc_segment_volume(voxelgrid, segment)

    with Pool() as pool:
        calculated_volumes = pool.map(calc_segment_volume_helper, print_segments)

    for segment, calc_vol in zip(print_segments, calculated_volumes):
        segment.calculated_volume = calc_vol
        segment.adjusted_volume = get_adjusted_segment_volume(
            parameters, segment, segment.calculated_volume
        )


def calc_analytical_volume(width: float, height: float, length: float) -> float:
    """Calculate the segment volume with constant parameters.

    Args:
        width (float): bead width
        height (float): bead height
        length (float): bead length

    Returns:
        float: bead volume
    """
    volume = width * height * pi * (1 / 4 * length + 1 / 6 * width)
    return volume


def get_adjusted_segment_volume(
    parameters: dict, segment: PrintSegment, calculated_volume: float
) -> float:
    """Constrain the given calclated volume for the print segment.

    Args:
        parameters (dict): dict of the model parameters.
        segment (PrintSegment): Segment to process.
        calculated_volume (float): Calculated volume.

    Returns:
        float: constrained volume
    """

    length = segment.length
    h_min = parameters["bead"]["height"]["minimal"]
    h_nom = parameters["bead"]["height"]["nominal"]
    w_max = parameters["bead"]["width"]["maximal"]
    w_min = parameters["bead"]["width"]["minimal"]
    w_nom = parameters["bead"]["width"]["nominal"]

    V_min = calc_analytical_volume(w_min, h_min, length)
    V_noN = calc_analytical_volume(w_max, h_nom, length)

    # Decide which value to use
    adjusted_volume: float = 0.0
    if calculated_volume < V_min:
        adjusted_volume = V_min
    elif abs(calculated_volume - V_noN) < 0.1:
        adjusted_volume = calc_analytical_volume(w_nom, h_nom, length)
    else:
        adjusted_volume = calculated_volume

    return adjusted_volume


def calc_segment_volume(voxelgrid: Voxels, segment: PrintSegment) -> float:
    """Calculate the volume of each segment by looking at all occupied voxels and summing up the segments share of volume of these voxels.
    Args:
        voxelgrid (Voxels): grid of voxels as given by binvox_rw.
        segment (PrintSegment): segment to be processed.

    Returns:
        float: The calculated volume.
    """
    if not segment.occupying_voxels_indices[0].any():
        return 0.0

    i, j, k = segment.occupying_voxels_indices
    data = voxelgrid.data[i, j, k]
    occupied_indices = np.where(data > 1)

    V_calc = np.sum(
        (voxelgrid.scale / voxelgrid.dims[0]) ** 3
        / np.subtract(data[occupied_indices], 1)
    )
    if segment.start_pose.position.as_array()[0] == 13.0018:
        breakpoint

    segment.num_shared_voxels += (data > 2).sum()
    return V_calc


@logtime
def write_adjusted_gcode(
    ostream, parameters: dict, pairs: Tuple[GcodeLine, PathSegment]
) -> None:
    """Exports the adjusted pairs of gcode and segments.

    Args:
        ostream (IO[Any]): output io.
        parameters (dict): The global parameters dict
        pairs (List[Tuple[GcodeParser.Gcode, PathSegment]]): The input pairs of gcode and derived path segments.
    """
    scaling_factor = 4 / (float(parameters["filament_diameter"]) ** 2 * pi)

    e_pos: float = 0.0
    for line, segment in pairs:
        if isinstance(segment, PrintSegment):
            volume = segment.adjusted_volume
            e_pos += volume * scaling_factor
            line.comment += f"Calculated Volume: {volume}"
        elif isinstance(segment, RetractionSegment):
            e_pos += segment.retraction_distance
        line.update_param("E", float(e_pos))
        ostream.write(line.gcode_str + "\n")


def registration(voxelgrid: Voxels, path_segments: List[PrintSegment]) -> np.ndarray:
    """Naive approach on registration. Assuming that the grid and the paths are only translated and not rotated.

    Args:
        voxelgrid (Voxels): grid of voxels as given by binvox_rw.
        path_segments (List[PathSegment]): Path segments

    Returns:
        np.ndarray: translation between the centers of both models.
    """

    upper_bound_voxel = np.max(
        voxelgrid.get_position_for_index(*np.where(voxelgrid.data > 0)), 1
    )
    lower_bound_voxel = np.min(
        voxelgrid.get_position_for_index(*np.where(voxelgrid.data > 0)), 1
    )

    positions = np.array(
        [segment.start_pose.position.as_array() for segment in path_segments]
    )
    upper_bound_gcode = np.max(positions.transpose(), 1)
    lower_bound_gcode = np.min(positions.transpose(), 1)
    lower_bound_gcode[2] = 0

    translation = (upper_bound_voxel - lower_bound_voxel) - (
        upper_bound_gcode - lower_bound_gcode
    )

    return translation
