#!/usr/bin/env python3
import unittest
from main import main


class TestBunny(unittest.TestCase):
    def test_bunny06(self):
        main(
            [
                "examples/bunny/bunny_0.6.yaml",
                "examples/bunny/bunny.stl",
                "examples/bunny/bunny_0.6.nc",
                "examples/bunny/bunny_0.6_out.nc",
            ]
        )

    def test_bunny04(self):
        main(
            [
                "examples/bunny/bunny_0.4.yaml",
                "examples/bunny/bunny.stl",
                "examples/bunny/bunny_0.4.nc",
                "examples/bunny/bunny_0.4_out.nc",
            ]
        )

    def test_bunny03(self):
        main(
            [
                "examples/bunny/bunny_0.3.yaml",
                "examples/bunny/bunny.stl",
                "examples/bunny/bunny_0.3.nc",
                "examples/bunny/bunny_0.3_out.nc",
            ]
        )

    def test_bunny02(self):
        main(
            [
                "examples/bunny/bunny_0.2.yaml",
                "examples/bunny/bunny.stl",
                "examples/bunny/bunny_0.2.nc",
                "examples/bunny/bunny_0.2_out.nc",
            ]
        )

    def test_bunny015(self):
        main(
            [
                "examples/bunny/bunny_0.15.yaml",
                "examples/bunny/bunny.stl",
                "examples/bunny/bunny_0.15.nc",
                "examples/bunny/bunny_0.15_out.nc",
            ]
        )

    def test_bunny01(self):
        main(
            [
                "examples/bunny/bunny_0.1.yaml",
                "examples/bunny/bunny.stl",
                "examples/bunny/bunny_0.1.nc",
                "examples/bunny/bunny_0.1_out.nc",
            ]
        )


if __name__ == "__main__":
    unittest.main()
