from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio.features
import stackstac
import xarray as xr
from scipy import constants

from .utils import _central_pixel_bbox, _compute_distance_to_center


def create(
    lat: Union[float, int],
    lon: Union[float, int],
    collection: str,
    start_date: str,
    end_date: str,
    bands: Optional[Union[str, List[str]]] = None,
    edge_size: Union[float, int] = 128.0,
    units: str = "px",
    resolution: Union[float, int] = 10.0,
    stac: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    gee: bool = False,
    stackstac_kw: Optional[dict] = None,
    gee_kw: Optional[dict] = None,
    **kwargs,
) -> xr.DataArray:
    """Creates a data cube from a STAC Catalogue as a :code:`xr.DataArray` object.

    The coordinates used here work as the approximate central coordinates of the data cube
    in the spatial dimension.

    Parameters
    ----------
    lat : float | int
        Latitude of the central pixel of the data cube.
    lon : float | int
        Longitude of the central pixel of the data cube.
    collection : str
        Name of the collection in the STAC Catalogue.
    start_date : str
        Start date of the data cube in YYYY-MM-DD format.
    end_date : str
        End date of the data cube in YYYY-MM-DD format.
    bands : str | List[str], default = None
        Name of the band(s) from the collection to use.
    edge_size : float | int, default = 128
        Size of the edge of the cube in the units specified by :code:`units`. All edges share the same size.

        .. warning::
           If :code:`edge_size` is not a multiple of 2, it will be rounded.

    units : str, default = 'px'
        Units of the provided edge size in :code:`edge_size`. Must be 'px' (pixels), 'm' (meters), or a unit
        name in https://docs.scipy.org/doc/scipy/reference/constants.html#units.

        .. versionadded:: 2024.1.1

        .. warning::
           Note that when :code:`units!='px'` the edge size will be transformed to meters (if :code:`units!='m'`).
           Furthermore, the edge size will be converted to pixels (using :code:`edge_size/resolution`)
           and rounded (see :code:`edge_size`). Therefore, the edge size when :code:`units!='px'` is just an approximation
           if its value in meters is not divisible by the requested resolution.

    resolution : float | int, default = 10
        Pixel size in meters.
    stac : str, default = 'https://planetarycomputer.microsoft.com/api/stac/v1'
        Endpoint of the STAC Catalogue to use.
    gee : bool, default = False
        Whether to use Google Earth Engine. This ignores the 'stac' argument.

        .. versionadded:: 2024.1.0

    stackstac_kw : dict, default = None
        Keyword arguments for :code:`stackstac` as a dictionary.

        .. versionadded:: 2024.1.0

    gee_kw : dict, default = None
        Keyword arguments for :code:`xarray.open_dataset()` with the :code:`xee` engine as a dictionary.
        Only used when :code:`gee=True`.

        .. versionadded:: 2026.10.0

        .. note::
           The pixel grid (:code:`crs`, :code:`crs_transform`, and :code:`shape_2d`) is defined by
           :code:`cubo` and cannot be passed here. Pixel coordinates are pixel centers, as given by :code:`xee`.

    kwargs :
        Additional keyword arguments passed to :code:`pystac_client.Client.search()`.

    Returns
    -------
    xr.DataArray
        Data Cube.


    Examples
    --------
    Create a Sentinel-2 L2A data cube with an edge size of 64 px from Planetary Computer:

    >>> import cubo
    >>> cubo.create(
    ...     lat=50,
    ...     lon=10,
    ...     collection="sentinel-2-l2a",
    ...     bands=["B02","B03","B04"],
    ...     start_date="2021-06-01",
    ...     end_date="2021-06-10",
    ...     edge_size=32,
    ...     resolution=10,
    ... )
    <xarray.DataArray (time: 3, band: 3, x: 32, y: 32)>

    Create a Sentinel-2 L2A data cube with an edge size of 128 px from Google Earth Engine:

    >>> import cubo
    >>> cubo.create(
    ...     lat=51.079225,
    ...     lon=10.452173,
    ...     collection="COPERNICUS/S2_SR_HARMONIZED",
    ...     bands=["B2","B3","B4"],
    ...     start_date="2016-06-01",
    ...     end_date="2017-07-01",
    ...     edge_size=128,
    ...     resolution=10,
    ...     gee=True,
    ... )
    <xarray.DataArray (time: 27, band: 3, x: 128, y: 128)>
    """
    # Harmonize units to pixels
    if units != "px":
        if units == "m":
            edge_size = edge_size / resolution
        else:
            edge_size = (edge_size * getattr(constants, units)) / resolution

    # Get the BBox and EPSG
    bbox_utm, bbox_latlon, utm_coords, epsg = _central_pixel_bbox(
        lat, lon, edge_size, resolution
    )

    # Use Google Earth Engine
    if gee:

        # Try to import ee, otherwise raise an ImportError
        try:
            import ee
            import xee
        except ImportError:
            raise ImportError(
                '"earthengine-api" and "xee" could not be loaded. Please install them, or install "cubo" using "pip install cubo[ee]"'
            )

        # Initialize Google Earth Engine with the high volume endpoint
        # User must do this before using cubo
        # ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")

        # Get BBox values in latlon
        west = bbox_latlon["coordinates"][0][0][0]
        south = bbox_latlon["coordinates"][0][0][1]
        east = bbox_latlon["coordinates"][0][2][0]
        north = bbox_latlon["coordinates"][0][2][1]

        # Create the BBox geometry in GEE
        BBox = ee.Geometry.BBox(west, south, east, north)

        # If the collection is string then access the Image Collection
        if isinstance(collection, str):
            collection = ee.ImageCollection(collection)

        # Do the filtering: Bounds, time, and bands
        collection = (
            collection.filterBounds(BBox).filterDate(start_date, end_date).select(bands)
        )

        # Get BBox values in UTM to define the pixel grid
        W = bbox_utm["coordinates"][0][0][0]
        E = bbox_utm["coordinates"][0][2][0]
        N = bbox_utm["coordinates"][0][2][1]
        S = bbox_utm["coordinates"][0][0][1]

        # Add xee arguments, chunks can be overridden
        gee_kw = {"chunks": dict(), **(gee_kw or dict())}

        # Return the cube via xee using the same pixel grid as stackstac
        cube = xr.open_dataset(
            collection,
            engine="ee",
            crs=f"EPSG:{epsg}",
            crs_transform=(resolution, 0, W, 0, -resolution, N),
            shape_2d=(round((E - W) / resolution), round((N - S) / resolution)),
            **gee_kw,
        )

        # Rearrange to match stackstac
        cube = cube.to_array("band").transpose("time", "band", "y", "x")

        # Delete all attributes
        cube.attrs = dict()

        # Get the name of the collection
        collection = collection.get("system:id").getInfo()

        # Override the stac argument using the GEE STAC
        stac = "https://earthengine-stac.storage.googleapis.com/catalog/catalog.json"

    else:

        # Convert UTM Bbox to a Feature
        bbox_utm = rasterio.features.bounds(bbox_utm)

        # Open the Catalogue
        CATALOG = pystac_client.Client.open(stac)

        # Do a search
        SEARCH = CATALOG.search(
            intersects=bbox_latlon,
            datetime=f"{start_date}/{end_date}",
            collections=[collection],
            **kwargs,
        )

        # Get all items and sign if using Planetary Computer
        items = SEARCH.item_collection()

        if stac == "https://planetarycomputer.microsoft.com/api/stac/v1":
            items = pc.sign(items)

        # Put the bands into list if not a list already
        if not isinstance(bands, list) and bands is not None:
            bands = [bands]

        # Add stackstac arguments
        if stackstac_kw is None:
            stackstac_kw = dict()

        # Create the cube
        cube = stackstac.stack(
            items,
            assets=bands,
            resolution=resolution,
            bounds=bbox_utm,
            epsg=epsg,
            **stackstac_kw,
        )

        # Delete attributes
        attributes = ["spec", "crs", "transform", "resolution"]

        for attribute in attributes:
            if attribute in cube.attrs:
                del cube.attrs[attribute]

    # Rounded edge size
    rounded_edge_size = cube.x.shape[0]

    # New attributes
    cube.attrs = dict(
        collection=collection,
        stac=stac,
        epsg=epsg,
        resolution=resolution,
        edge_size=rounded_edge_size,
        edge_size_m=rounded_edge_size * resolution,
        central_lat=lat,
        central_lon=lon,
        central_y=utm_coords[1],
        central_x=utm_coords[0],
        time_coverage_start=start_date,
        time_coverage_end=end_date,
    )

    cube = cube.assign_coords(
        {"cubo:distance_from_center": (["y", "x"], _compute_distance_to_center(cube))}
    )

    # New name
    cube.name = collection

    return cube
