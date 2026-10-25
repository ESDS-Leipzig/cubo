import unittest

import numpy as np
import pandas as pd
import xarray as xr

import cubo
from cubo.utils import _central_pixel_bbox, _compute_distance_to_center


class TestUtils(unittest.TestCase):
    """Offline tests for the cubo utilities."""

    def test_central_pixel_bbox(self):
        """Test the BBox, UTM coordinates and EPSG"""
        bbox_utm, bbox_latlon, utm_coords, epsg = _central_pixel_bbox(
            lat=50, lon=10, edge_size=32, resolution=10
        )
        self.assertEqual(epsg, 32632)
        self.assertEqual(len(utm_coords), 2)

        # The polygon must be closed and the edges must be edge_size * resolution
        polygon = np.array(bbox_utm["coordinates"][0])
        self.assertEqual(polygon.shape, (5, 2))
        np.testing.assert_array_equal(polygon[0], polygon[-1])
        self.assertEqual(polygon[:, 0].max() - polygon[:, 0].min(), 320)
        self.assertEqual(polygon[:, 1].max() - polygon[:, 1].min(), 320)

        # The latlon BBox must contain the central coordinates
        polygon_latlon = np.array(bbox_latlon["coordinates"][0])
        self.assertTrue(polygon_latlon[:, 0].min() < 10 < polygon_latlon[:, 0].max())
        self.assertTrue(polygon_latlon[:, 1].min() < 50 < polygon_latlon[:, 1].max())

    def test_central_pixel_bbox_southern_hemisphere(self):
        """Test the EPSG for the southern hemisphere"""
        *_, epsg = _central_pixel_bbox(lat=-10, lon=-75, edge_size=32, resolution=10)
        self.assertEqual(epsg, 32718)

    def test_compute_distance_to_center(self):
        """Test the distance to the center"""
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["y", "x"],
            coords=dict(y=[20.0, 10.0, 0.0], x=[0.0, 10.0, 20.0, 30.0]),
            attrs=dict(central_x=0.0, central_y=0.0),
        )
        distance = _compute_distance_to_center(da)
        self.assertEqual(distance.shape, (3, 4))
        self.assertEqual(distance[2, 0], 0.0)
        self.assertAlmostEqual(distance[0, 3], np.hypot(30, 20))


class Test(unittest.TestCase):
    """Tests for the cubo package."""

    def test_planetary_computer(self):
        """Test the cubo"""
        da = cubo.create(
            lat=50,
            lon=10,
            collection="sentinel-2-l2a",
            bands=["B02", "B03", "B04"],
            start_date="2021-06-01",
            end_date="2021-06-10",
            edge_size=32,
            resolution=10,
        )
        self.assertIsInstance(da, xr.DataArray)
        self.assertEqual(da.dims, ("time", "band", "y", "x"))
        self.assertEqual(da.sizes["x"], 32)
        self.assertEqual(da.sizes["y"], 32)
        self.assertGreater(da.sizes["time"], 0)
        self.assertEqual(list(da.band.values), ["B02", "B03", "B04"])
        self.assertEqual(da.name, "sentinel-2-l2a")
        self.assertIn("cubo:distance_from_center", da.coords)

        # Distance from center must match the pixel coordinates
        x, y = np.meshgrid(da.x, da.y)
        np.testing.assert_allclose(
            da["cubo:distance_from_center"].values,
            np.hypot(x - da.attrs["central_x"], y - da.attrs["central_y"]),
        )

        # Attributes
        self.assertEqual(da.attrs["epsg"], 32632)
        self.assertEqual(da.attrs["edge_size"], 32)
        self.assertEqual(da.attrs["edge_size_m"], 320)
        self.assertEqual(da.attrs["central_lat"], 50)
        self.assertEqual(da.attrs["central_lon"], 10)

        # Time range
        self.assertTrue((da.time >= pd.Timestamp("2021-06-01")).all())
        self.assertTrue((da.time <= pd.Timestamp("2021-06-11")).all())

    def test_single_band(self):
        """Test the cubo with a single band as a string"""
        da = cubo.create(
            lat=50,
            lon=10,
            collection="sentinel-2-l2a",
            bands="B04",
            start_date="2021-06-01",
            end_date="2021-06-10",
            edge_size=32,
            resolution=10,
        )
        self.assertEqual(list(da.band.values), ["B04"])

    def test_units(self):
        """Test the cubo with the edge size in meters and kilometers"""
        for edge_size, units in [(320, "m"), (0.32, "kilo")]:
            with self.subTest(units=units):
                da = cubo.create(
                    lat=50,
                    lon=10,
                    collection="sentinel-2-l2a",
                    bands=["B02", "B03", "B04"],
                    start_date="2021-06-01",
                    end_date="2021-06-10",
                    edge_size=edge_size,
                    units=units,
                    resolution=10,
                )
                self.assertEqual(da.sizes["x"], 32)
                self.assertEqual(da.sizes["y"], 32)

    def test_stackstac(self):
        """Test the cubo with stackstac kwargs"""
        da = cubo.create(
            lat=50,
            lon=10,
            collection="sentinel-2-l2a",
            bands=["B02", "B03", "B04"],
            start_date="2021-06-01",
            end_date="2021-06-10",
            edge_size=32,
            resolution=10,
            stackstac_kw=dict(xy_coords="center"),
        )
        self.assertIsInstance(da, xr.DataArray)
        self.assertEqual(da.sizes["x"], 32)

    # def test_element84(self):
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
