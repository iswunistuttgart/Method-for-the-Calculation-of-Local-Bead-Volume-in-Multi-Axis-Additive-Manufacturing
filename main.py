#!/usr/bin/env python3
import argparse
import logging
import sys
import yaml
import numpy as np
from PathSegment import PrintSegment
from Voxelizer import CudaVoxelizer as Voxelizer
from typing import List

from processing import (
    parse_gcode,
    offset_path_segments,
    interpolate_path_segments,
    trace_paths,
    calc_volume,
    write_adjusted_gcode,
    registration,
)

from util import write_report
from plotting import plot_histogram, parallel_plot_path_voxels, plot_paths


def get_parser():
    parser = argparse.ArgumentParser(
        prog="Standalone volumetric postprocessor",
        description="Parses existing gcode for 3d printers, calculates local volume and re-rexports gcode with local volumes",
    )
    parser.add_argument("parameter_file")
    parser.add_argument("model")
    parser.add_argument("input_nc")
    parser.add_argument("output_nc")
    parser.add_argument('--report', action='store_true')
    parser.add_argument("--plot", nargs="+", choices= ['hist', 'lines', 'voxels'])
    return parser


def main(argv=None):
    logger = logging.getLogger("volume calculator")
    logger.setLevel(logging.INFO)

    # Make info level log lines print to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    parser = get_parser()
    args = parser.parse_args(argv)

    # Read parameter file
    with open(args.parameter_file, "r") as pfile:
        parameters = yaml.safe_load(pfile)

    # Read input nc file to buffer
    with open(args.input_nc, "r") as input_nc_file:
        input_gcode = input_nc_file.read()

    # Process mesh
    logger.info(f"Voxelizing mesh: {args.model}")
    voxelizer = Voxelizer(parameters)
    voxelgrid = voxelizer.voxelize(args.model)
    logger.info(f"Grid dimensions: {voxelgrid.dims[0]}")

    logger.info(f"Parsing gcode: {args.input_nc}")
    pairs = parse_gcode(parameters, input_gcode)

    if "interpolate_length" in parameters:
        logger.info(
            f"Re-discretizing segments to a maximal length of {parameters['interpolate_length']} mm"
        )
        pairs = interpolate_path_segments(parameters, pairs)

    # get references to all relevant (printing_type) segments
    print_segments: List[PrintSegment] = [seg for _, seg in pairs if isinstance(seg, PrintSegment)]

    logger.info("Post-processing paths.")
    offset_path_segments(print_segments, float(parameters["bead"]["height"]["nominal"]))

    # Registration must be applied, when the slicer has translated the model from the original location
    if "registration" in parameters:
        translation = None
        if isinstance(parameters["registration"], list):
            translation = np.array(parameters["registration"])
        elif (
            isinstance(parameters["registration"], bool) and parameters["registration"]
        ):
            translation = registration(voxelgrid, print_segments)

        if translation is not None:
            voxelgrid.translate -= translation
            logger.info(f"Applying translation {translation}")

    logger.info("Step 1: Trace paths.")
    trace_paths(parameters, voxelgrid, print_segments)

    logger.info("Step 2: Calculate sum of voxel volumes for each segment.")
    calc_volume(parameters, voxelgrid, print_segments)

    logger.info("Writing adjusted gcode.")
    with open(args.output_nc, "w") as output_nc_stream:
        write_adjusted_gcode(output_nc_stream, parameters, pairs)

    if args.report:
        logger.info("Writing report")
        write_report(args.input_nc + ".report.csv", pairs, voxelgrid)

    if args.plot:
        logger.info("Generating plots")

        if 'lines' in args.plot:
            # Paths with coloring
            logger.info("Plotting line graph with colors.")
            logger.info("Note: This is only sensible for single-layer models.")
            plot_paths(parameters, print_segments)

        if 'voxels' in args.plot:
            logger.info("Plotting voxel volumes.") 
            logger.info("Note: This is only sensible for single-layer models.")
            # Voxelgrid colored by the paths' cross-sectional area
            parallel_plot_path_voxels(parameters, print_segments, voxelgrid)

        if 'hist' in args.plot:
            # Distribution of cross-sections
            logger.info("Plogtting histogram.")
            plot_histogram(print_segments)



if __name__ == "__main__":
    sys.exit(main())
