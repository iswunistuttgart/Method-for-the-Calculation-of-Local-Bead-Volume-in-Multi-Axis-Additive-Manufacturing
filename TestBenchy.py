#!/usr/bin/env python3
import unittest
from main import main


class TestBenchy(unittest.TestCase):
    def test_benchy01(self):
        main(
            [
                "examples/benchy/benchy_0.1.yaml",
                "examples/benchy/benchy.stl",
                "examples/benchy/benchy_0.1.nc",
                "examples/benchy/benchy_0.1_out.nc",
            ]
        )

    def test_benchy015(self):
        main(
            [
                "examples/benchy/benchy_0.15.yaml",
                "examples/benchy/benchy.stl",
                "examples/benchy/benchy_0.15.nc",
                "examples/benchy/benchy_0.15_out.nc",
            ]
        )

    def test_benchy02(self):
        main(
            [
                "examples/benchy/benchy_0.2.yaml",
                "examples/benchy/benchy.stl",
                "examples/benchy/benchy_0.2.nc",
                "examples/benchy/benchy_0.2_out.nc",
            ]
        )

    def test_benchy03(self):
        main(
            [
                "examples/benchy/benchy_0.3.yaml",
                "examples/benchy/benchy.stl",
                "examples/benchy/benchy_0.3.nc",
                "examples/benchy/benchy_0.3_out.nc",
            ]
        )

    def test_benchy04(self):
        main(
            [
                "examples/benchy/benchy_0.4.yaml",
                "examples/benchy/benchy.stl",
                "examples/benchy/benchy_0.4.nc",
                "examples/benchy/benchy_0.4_out.nc",
            ]
        )

    def test_benchy05(self):
        main(
            [
                "examples/benchy/benchy_0.5.yaml",
                "examples/benchy/benchy.stl",
                "examples/benchy/benchy_0.5.nc",
                "examples/benchy/benchy_0.5_out.nc",
            ]
        )

    def test_benchy06(self):
        main(
            [
                "examples/benchy/benchy_0.6.yaml",
                "examples/benchy/benchy.stl",
                "examples/benchy/benchy_0.6.nc",
                "examples/benchy/benchy_0.6_out.nc",
            ]
        )


if __name__ == "__main__":
    unittest.main()
