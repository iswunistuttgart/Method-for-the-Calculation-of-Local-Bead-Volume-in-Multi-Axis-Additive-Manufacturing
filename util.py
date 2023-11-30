from functools import wraps
import time
import logging
from PathSegment import PrintSegment
from typing import Tuple, List
from gcodeparser import GcodeLine
from PathSegment import PathSegment
from binvox_rw import Voxels


def logtime(func):
    """Wrapper that logs the real time, that the associated funtion takes for processing.
    """
    def get_logger():
        logger = logging.getLogger("Timer")
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        return logger

    @wraps(func)
    def logtime_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total = end_time - start_time
        logger = get_logger()
        logger.info(f"{func.__name__} execution took {total:.4f} seconds")
        return result

    return logtime_wrapper


def write_report(filename: str, pairs: List[Tuple[GcodeLine, PathSegment]], voxelgrid: Voxels) -> None:
    """Write a report on the given, processed data.

    Args:
        filename (str): Path to the report file. Must be writable.
        pairs (List[Tuple[GcodeLine, PathSegment]]): -
        voxelgrid (Voxels): _description_
    """
    original_volume: float = 0.0
    adjusted_volume: float = 0.0
    calculated_volume: float = 0.0
    total_length: float  = 0.0

    print_segments = []
    for _, seg in pairs:
        if isinstance(seg, PrintSegment):
            print_segments.append(seg)

    for seg in print_segments:
        original_volume += seg.slicer_volume
        adjusted_volume += seg.adjusted_volume
        calculated_volume += seg.calculated_volume
        total_length += seg.length

    with open(filename, "w") as report:
        report.write(
            f"# Printing Segments:         {len(print_segments)}\n"
            f"1 Voxel Edge Length:         {(voxelgrid.scale/voxelgrid.dims[0])}mm\n"
            f"1 Voxel Volume:              {(voxelgrid.scale/voxelgrid.dims[0])**3}mm^3\n"
            f"Printing paths total length: {total_length}\n"
            f"Total volume from voxels:    {(voxelgrid.scale/voxelgrid.dims[0])**3*(voxelgrid.data > 0).sum()}\n"
            f"Volume original g-code:      {original_volume}\n"
            f"Volume calculated g-code:    {calculated_volume}\n"
            f"Volume adjusted g-code:      {adjusted_volume}\n"
            f"\n\n#### CSV Segment values ####\n"
            f"G-Code line;segment length;original volume;calculated volume; adjusted volume;# voxels total;# shared voxels;difference calculator/slicer;deviation calculator/slicer [%]; volume/length original; volume/length calculated; volume/length adjusted\n"
        )

        for i, (_, seg) in enumerate(pairs):
            if isinstance(seg, PrintSegment):
                report.write(
                    f"{i};{seg.length:.2f};{seg.slicer_volume:.2f};{seg.calculated_volume:.2f};{seg.adjusted_volume:.2f};{len(seg.occupying_voxels_indices[0])};{seg.num_shared_voxels};{(seg.calculated_volume - seg.slicer_volume):.2f};{(seg.calculated_volume - seg.slicer_volume)*100/seg.slicer_volume:.2f};{seg.slicer_volume / seg.length:.2f};{seg.calculated_volume /seg.length:.2f};{seg.adjusted_volume/seg.length:.2f}\n".replace(
                        ".", ","
                    )
                )
