import unittest

import numpy as np
import pandas as pd
import xarray as xr

import cubo


class Test(unittest.TestCase):
    """Tests for the cubo package."""

    def test_planetary_computer(self):
        """Test the cubo"""
        da = cubo.create(
            lat=50,
            lon=10,
            collection="sentinel-2-l2a",
            bands=["B02","B03","B04"],
            start_date="2021-06-01",
            end_date="2021-06-10",
            edge_size=32,
            resolution=10,
        )
        self.assertIsInstance(da, xr.DataArray)

    #def test_element84(self):
    #    """Test the cubo"""
    #    da = cubo.create(
    #        lat=50,
    #        lon=10,
    #        collection="sentinel-s2-l2a-cogs",
    #        bands=["B02","B03","B04"],
    #        start_date="2021-06-01",
    #        end_date="2021-06-10",
    #        edge_size=32,
    #        resolution=10,
    #        stac="https://earth-search.aws.element84.com/v0"
    #    )
    #    self.assertIsInstance(da, xr.DataArray)


if __name__ == "__main__":
    unittest.main()
