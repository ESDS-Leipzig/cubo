import unittest

import numpy as np
import pandas as pd
import xarray as xr

import cubo


class Test(unittest.TestCase):
    """Tests for the cubo package."""

    def test_0(self):
        """Test the cubo"""
        self.assertIsInstance(1, int)


if __name__ == "__main__":
    unittest.main()
