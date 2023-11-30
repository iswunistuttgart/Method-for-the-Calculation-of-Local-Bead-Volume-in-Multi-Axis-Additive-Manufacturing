#!/usr/bin/env python3
import unittest
from main import main


class TestBoxWithHole(unittest.TestCase):
    def test_test_box_with_hole(self):
        main(
            [
                "examples/box_with_hole_iterative/box_with_hole_iterative_0.5.yaml",
                "examples/box_with_hole_iterative/box_with_hole_iterative.stl",
                "examples/box_with_hole_iterative/box_with_hole_iterative_0.5.nc",
                "examples/box_with_hole_iterative/box_with_hole_iterative_0.5_out.nc",
            ]
        )


if __name__ == "__main__":
    unittest.main()
