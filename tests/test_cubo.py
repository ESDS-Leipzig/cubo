import unittest

import numpy as np
import pandas as pd
import xarray as xr

import cubo
from cubo.cubo import _merge
from cubo.utils import (
    _central_pixel_bbox,
    _compute_distance_to_center,
    _gee_end_date,
    _parse_collections,
    _split_kw,
    _split_search_kw,
)

PC = "https://planetarycomputer.microsoft.com/api/stac/v1"


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

    def test_parse_single_collection(self):
        """Test a single collection with a single band"""
        requests = _parse_collections("sentinel-2-l2a", PC, "B04", False)
        self.assertEqual(
            requests, [dict(collection="sentinel-2-l2a", stac=PC, bands=["B04"])]
        )

    def test_parse_gee(self):
        """Test that gee=True overrides the STAC endpoint"""
        requests = _parse_collections("COPERNICUS/S2_SR_HARMONIZED", PC, None, True)
        self.assertEqual(requests[0]["stac"], "ee")
        self.assertIsNone(requests[0]["bands"])

    def test_parse_multiple_collections(self):
        """Test multiple collections from multiple STAC endpoints"""
        requests = _parse_collections(
            ["sentinel-2-l2a", "sentinel-1-rtc", "COPERNICUS/S2_CLOUD_PROBABILITY"],
            {
                PC: ["sentinel-2-l2a", "sentinel-1-rtc"],
                "ee": "COPERNICUS/S2_CLOUD_PROBABILITY",
            },
            {
                "sentinel-2-l2a": ["B04", "B08"],
                "COPERNICUS/S2_CLOUD_PROBABILITY": "probability",
            },
            False,
        )
        self.assertEqual([r["stac"] for r in requests], [PC, PC, "ee"])
        self.assertEqual(
            [r["bands"] for r in requests], [["B04", "B08"], None, ["probability"]]
        )

    def test_parse_errors(self):
        """Test the validation of collections, STAC endpoints and bands"""
        collections = ["sentinel-2-l2a", "sentinel-1-rtc"]
        invalid = [
            (ValueError, dict(collection=[])),
            (ValueError, dict(collection=["sentinel-2-l2a", "sentinel-2-l2a"])),
            (TypeError, dict(bands=["B04"])),
            (ValueError, dict(stac={PC: "sentinel-2-l2a"})),
            (ValueError, dict(stac={PC: collections + ["landsat-c2-l2"]})),
            (ValueError, dict(stac={PC: collections, "ee": "sentinel-2-l2a"})),
            (ValueError, dict(bands={"landsat-c2-l2": "red"})),
        ]
        for error, kw in invalid:
            with self.subTest(kw=kw):
                kw = dict(dict(collection=collections, stac=PC, bands=None), **kw)
                with self.assertRaises(error):
                    _parse_collections(gee=False, **kw)

    def test_split_kw(self):
        """Test global and per-collection keyword arguments"""
        collections = ["sentinel-2-l2a", "sentinel-1-rtc"]
        self.assertEqual(_split_kw(None, collections), [dict(), dict()])
        self.assertEqual(
            _split_kw(dict(dtype="float32"), collections),
            [dict(dtype="float32"), dict(dtype="float32")],
        )
        self.assertEqual(
            _split_kw({"sentinel-1-rtc": dict(dtype="float32")}, collections),
            [dict(), dict(dtype="float32")],
        )

    def test_split_search_kw(self):
        """Test global and per-collection search keyword arguments"""
        collections = ["sentinel-2-l2a", "sentinel-1-rtc"]
        query = {"eo:cloud_cover": {"lt": 20}}
        self.assertEqual(
            _split_search_kw(dict(query=query, limit=10), collections),
            [dict(query=query, limit=10), dict(query=query, limit=10)],
        )
        self.assertEqual(
            _split_search_kw(dict(query={"sentinel-2-l2a": query}), collections),
            [dict(query=query), dict()],
        )

    def test_gee_end_date(self):
        """Test the exclusive end date for Google Earth Engine"""
        self.assertEqual(_gee_end_date("2021-06-10"), "2021-06-11T00:00:00")
        self.assertEqual(
            _gee_end_date("2021-06-10T12:00:00"), "2021-06-10T12:00:00.001000"
        )

    def test_merge(self):
        """Test merging cubes with duplicated and shared timestamps"""

        def cube(name, times, bands):
            times = pd.to_datetime(times)
            da = xr.DataArray(
                np.ones((len(times), len(bands), 2, 2)),
                dims=["time", "band", "y", "x"],
                coords=dict(time=times, band=bands, y=[10.0, 0.0], x=[0.0, 10.0]),
                attrs=dict(collection=name, stac="ee"),
                name=name,
            )
            return da.assign_coords(
                {"cubo:distance_from_center": (["y", "x"], np.zeros((2, 2)))}
            )

        a = cube("a", ["2021-06-01", "2021-06-01", "2021-06-03"], ["B1"])
        b = cube("b", ["2021-06-01", "2021-06-02"], ["B2", "B3"])
        da = _merge([a, b])

        self.assertEqual(list(da.band.values), ["B1", "B2", "B3"])
        self.assertEqual(list(da.collection.values), ["a", "b", "b"])
        self.assertEqual(
            list(da.time.values),
            list(
                pd.to_datetime(["2021-06-01", "2021-06-01", "2021-06-02", "2021-06-03"])
            ),
        )

        # The first 2021-06-01 is shared, the second one only exists in "a"
        np.testing.assert_array_equal(
            da.isel(y=0, x=0).notnull().values,
            [
                [True, True, True],
                [True, False, False],
                [False, True, True],
                [True, False, False],
            ],
        )
        self.assertEqual(da.attrs["collection"], ["a", "b"])

        with self.assertRaises(ValueError):
            _merge([a, cube("c", ["2021-06-01"], ["B1"])])

    def test_create_errors(self):
        """Test the validation of cubo.create() before requesting any data"""
        kw = dict(
            lat=50,
            lon=10,
            collection="sentinel-2-l2a",
            start_date="2021-06-01",
            end_date="2021-06-10",
        )
        with self.assertRaises(ValueError):
            cubo.create(xy_coords="corner", **kw)
        with self.assertRaises(ValueError):
            cubo.create(stackstac_kw=dict(xy_coords="center"), **kw)


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

        # Distance from center must match the pixel centers (coordinates are top-left)
        x, y = np.meshgrid(da.x + 5, da.y - 5)
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
        self.assertTrue((da.time < pd.Timestamp("2021-06-11")).all())
        self.assertTrue((da.time >= pd.Timestamp("2021-06-10")).any())

    def test_xy_coords(self):
        """Test the cubo with the coordinates at the center of the pixels"""
        kw = dict(
            lat=50,
            lon=10,
            collection="sentinel-2-l2a",
            bands="B04",
            start_date="2021-06-01",
            end_date="2021-06-10",
            edge_size=32,
            resolution=10,
        )
        topleft = cubo.create(**kw)
        center = cubo.create(xy_coords="center", **kw)
        np.testing.assert_allclose(center.x - topleft.x, 5)
        np.testing.assert_allclose(center.y - topleft.y, -5)
        np.testing.assert_allclose(
            center["cubo:distance_from_center"], topleft["cubo:distance_from_center"]
        )

    def test_dataset(self):
        """Test the cubo as a dataset"""
        ds = cubo.create(
            lat=50,
            lon=10,
            collection="sentinel-2-l2a",
            bands=["B02", "B03", "B04"],
            start_date="2021-06-01",
            end_date="2021-06-10",
            edge_size=32,
            resolution=10,
            dataset=True,
        )
        self.assertIsInstance(ds, xr.Dataset)
        self.assertEqual(list(ds.data_vars), ["B02", "B03", "B04"])
        self.assertEqual(ds.attrs["collection"], "sentinel-2-l2a")

    def test_multiple_collections(self):
        """Test the cubo with multiple collections merged"""
        kw = dict(
            lat=50,
            lon=10,
            collection=["sentinel-2-l2a", "sentinel-1-rtc"],
            bands={"sentinel-2-l2a": ["B04", "B08"], "sentinel-1-rtc": ["vv", "vh"]},
            start_date="2021-06-01",
            end_date="2021-06-10",
            edge_size=32,
            resolution=10,
        )
        da = cubo.create(**kw)
        self.assertIsInstance(da, xr.DataArray)
        self.assertEqual(list(da.band.values), ["B04", "B08", "vv", "vh"])
        self.assertEqual(
            list(da.collection.values),
            ["sentinel-2-l2a", "sentinel-2-l2a", "sentinel-1-rtc", "sentinel-1-rtc"],
        )
        self.assertEqual(da.attrs["collection"], ["sentinel-2-l2a", "sentinel-1-rtc"])
        self.assertIn("cubo:distance_from_center", da.coords)
        self.assertNotIn("s2:mgrs_tile", da.coords)

        ds = cubo.create(dataset=True, **kw)
        self.assertEqual(list(ds.data_vars), ["B04", "B08", "vv", "vh"])
        self.assertEqual(ds["vv"].attrs["collection"], "sentinel-1-rtc")

    def test_datatree(self):
        """Test the cubo with multiple collections as a data tree"""
        kw = dict(
            lat=50,
            lon=10,
            collection=["sentinel-2-l2a", "sentinel-1-rtc"],
            bands={"sentinel-2-l2a": ["B04", "B08"], "sentinel-1-rtc": "vv"},
            stac={PC: ["sentinel-2-l2a", "sentinel-1-rtc"]},
            start_date="2021-06-01",
            end_date="2021-06-10",
            edge_size=32,
            resolution=10,
            datatree=True,
            query={"sentinel-2-l2a": {"eo:cloud_cover": {"lt": 100}}},
        )
        dt = cubo.create(**kw)
        self.assertIsInstance(dt, xr.DataTree)
        self.assertEqual(set(dt.children), {"sentinel-2-l2a", "sentinel-1-rtc"})
        self.assertEqual(dt["sentinel-2-l2a"]["data"].dims, ("time", "band", "y", "x"))
        self.assertEqual(list(dt["sentinel-1-rtc"]["data"].band.values), ["vv"])
        self.assertEqual(set(dt.coords), {"y", "x", "cubo:distance_from_center"})
        self.assertNotIn("s2:mgrs_tile", dt["sentinel-1-rtc"].coords)
        np.testing.assert_array_equal(dt["sentinel-2-l2a"].x, dt["sentinel-1-rtc"].x)
        self.assertEqual(dt["sentinel-1-rtc"].attrs["collection"], "sentinel-1-rtc")

        dt = cubo.create(dataset=True, **kw)
        self.assertEqual(list(dt["sentinel-2-l2a"].data_vars), ["B04", "B08"])

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
            stackstac_kw=dict(
                dtype="float32", fill_value=np.float32(np.nan), rescale=False
            ),
        )
        self.assertIsInstance(da, xr.DataArray)
        self.assertEqual(da.dtype, np.float32)
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
