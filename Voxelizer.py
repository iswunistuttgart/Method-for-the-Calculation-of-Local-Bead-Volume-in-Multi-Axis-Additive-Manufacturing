import os
import subprocess
import numpy as np
import logging
import binvox_rw
import trimesh


class CudaVoxelizer:
    cuda_voxelizer_binary = os.getcwd() + "/external/cuda_voxelizer"

    def __init__(self, paramters):
        self.logger = logging.getLogger("CudaVoxelizer")
        self.voxel_size : float = paramters["voxelsize"]

    def calc_voxelgrid_dimension(self, mesh: trimesh.Trimesh, voxel_size : float) -> int:
        AABB = mesh.bounds

        max_distance_axis_aligned = np.max(np.abs(mesh.bounds[0] - mesh.bounds[1]))

        # find a power of 2 dimension, to determine the grid size
        minimal_grid_dim = max_distance_axis_aligned / voxel_size
        pow = 2
        while pow < minimal_grid_dim:
            pow *= 2
        return pow

    def is_cached(self, file, cuda_voxel_number) -> bool:
        return os.path.isfile(self.expected_cuda_output(file, cuda_voxel_number))

    def expected_cuda_output(self, stl_file, cuda_voxel_number) -> str:
        return stl_file + "_" + str(cuda_voxel_number) + ".binvox"

    def voxelize(self, input_file) -> binvox_rw.Voxels:
        if not os.path.isfile(input_file):
            raise ValueError("stl_file does not exist.")

        if not os.path.isfile(self.cuda_voxelizer_binary):
            raise ValueError("cuda_voxelizer not found.")

        part_mesh = trimesh.load_mesh(input_file)

        # calc the voxel resolution
        cuda_voxel_number = self.calc_voxelgrid_dimension(part_mesh, self.voxel_size)

        if not self.is_cached(input_file, cuda_voxel_number):
            # voxelize full part using cuda_voxelizer
            # TODO: Check for watertight mesh first, otherwise this might output unexpected results
            # FIXME: Solid voxelization with cuda_voxelizer is marked as experimental
            cmd = [
                self.cuda_voxelizer_binary,
                "-f",
                input_file,
                "-s",
                str(cuda_voxel_number),
                "-o",
                "binvox",
                "-solid",
            ]
            self.logger.debug("Calling cuda_voxelizer: {*cmd}")
            process = subprocess.Popen(cmd, stdout=None)
            process.wait()
            self.logger.debug("process exited.")
            if not self.is_cached(input_file, cuda_voxel_number):
                raise Exception(
                    f"file {self.expected_cuda_output(input_file, cuda_voxel_number)} does not exist after voxelization."
                )
        else:
            self.logger.info("Cache hit! Skipping voxelization.")

        filename = self.expected_cuda_output(input_file, cuda_voxel_number)

        with open(filename, "rb") as f:
            grid = binvox_rw.read_as_3d_array(f, fix_coords=True)

        return grid
