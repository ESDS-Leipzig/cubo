from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio.features
import stackstac
import xarray as xr
from scipy import constants

from .utils import (
    _central_pixel_bbox,
    _compute_distance_to_center,
    _gee_end_date,
    _parse_collections,
    _split_kw,
    _split_search_kw,
)


def create(
    lat: Union[float, int],
    lon: Union[float, int],
    collection: Union[str, List[str]],
    start_date: str,
    end_date: str,
    bands: Optional[Union[str, List[str], dict]] = None,
    edge_size: Union[float, int] = 128.0,
    units: str = "px",
    resolution: Union[float, int] = 10.0,
    stac: Union[str, dict] = "https://planetarycomputer.microsoft.com/api/stac/v1",
    gee: bool = False,
    stackstac_kw: Optional[dict] = None,
    gee_kw: Optional[dict] = None,
    xy_coords: str = "topleft",
    dataset: bool = False,
    datatree: bool = False,
    **kwargs,
) -> Union[xr.DataArray, xr.Dataset, xr.DataTree]:
    """Creates a data cube from one or multiple collections of STAC Catalogues and/or Google Earth Engine.

    The coordinates used here work as the approximate central coordinates of the data cube
    in the spatial dimension. All collections share the same pixel grid.

    Parameters
    ----------
    lat : float | int
        Latitude of the central pixel of the data cube.
    lon : float | int
        Longitude of the central pixel of the data cube.
    collection : str | List[str]
        Name of the collection(s) to use. When using Google Earth Engine, a single
        :code:`ee.ImageCollection` object can also be used.

        .. versionchanged:: 2026.10.0
           Multiple collections can be requested as a list.

    start_date : str
        Start date of the data cube in YYYY-MM-DD format (included).
    end_date : str
        End date of the data cube in YYYY-MM-DD format (included).

        .. versionchanged:: 2026.10.0
           The end date is included for both STAC Catalogues and Google Earth Engine.

    bands : str | List[str] | dict, default = None
        Name of the band(s) to use. If None, all bands are used. When requesting multiple collections,
        it must be a dictionary with collections as keys and the band(s) of each collection as values
        (collections not in the dictionary use all bands).

        .. versionchanged:: 2026.10.0
           Bands can be a dictionary with the band(s) of each collection.

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
        Pixel size in meters. Shared by all collections.
    stac : str | dict, default = 'https://planetarycomputer.microsoft.com/api/stac/v1'
        Endpoint of the STAC Catalogue to use, or 'ee' to use Google Earth Engine. When the collections come
        from different catalogues, it must be a dictionary with endpoints as keys and the collection(s) of
        each endpoint as values (use the 'ee' key for Google Earth Engine collections).

        .. versionchanged:: 2026.10.0
           The STAC endpoint can be a dictionary, and 'ee' can be used for Google Earth Engine.

    gee : bool, default = False
        Whether to use Google Earth Engine for all collections. This ignores the 'stac' argument.

        .. versionadded:: 2024.1.0

    stackstac_kw : dict, default = None
        Keyword arguments for :code:`stackstac` as a dictionary. To use different keyword arguments
        per collection, use a dictionary with collections as keys and the keyword arguments of each
        collection as values.

        .. versionadded:: 2024.1.0

        .. note::
           :code:`xy_coords` cannot be passed here, use the :code:`xy_coords` argument instead.

    gee_kw : dict, default = None
        Keyword arguments for :code:`xarray.open_dataset()` with the :code:`xee` engine as a dictionary.
        Only used for Google Earth Engine collections. To use different keyword arguments per collection,
        use a dictionary with collections as keys and the keyword arguments of each collection as values.

        .. versionadded:: 2026.10.0

        .. note::
           The pixel grid (:code:`crs`, :code:`crs_transform`, and :code:`shape_2d`) is defined by
           :code:`cubo` and cannot be passed here.

    xy_coords : str, default = 'topleft'
        Whether the x and y coordinates are the 'topleft' corners or the 'center' of the pixels. The
        :code:`cubo:distance_from_center` coordinate is always computed from the center of the pixels.

        .. versionadded:: 2026.10.0

    dataset : bool, default = False
        Whether to return each cube as a :code:`xr.Dataset` with one variable per band instead of a
        :code:`xr.DataArray`.

        .. versionadded:: 2026.10.0

    datatree : bool, default = False
        Whether to return a :code:`xr.DataTree` with one node per collection. Each node keeps its own
        time and band dimensions, while the spatial coordinates are shared. Each node contains the cube
        as the 'data' variable (or one variable per band if :code:`dataset=True`). Collections with '/'
        in their name (e.g., Google Earth Engine collections) are stored as nested nodes.

        .. versionadded:: 2026.10.0

    kwargs :
        Additional keyword arguments passed to :code:`pystac_client.Client.search()`. To use a different
        value per collection, use a dictionary with collections as keys as the value of the keyword argument.

    Returns
    -------
    xr.DataArray | xr.Dataset | xr.DataTree
        Data Cube. When requesting multiple collections without :code:`datatree=True`, the cubes are
        merged along the band dimension and the time dimension is the union of the timestamps of all
        collections (with NaN where a collection has no data). Metadata coordinates of each collection
        are dropped and a :code:`collection` coordinate is added along the band dimension.

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
    <xarray.DataArray (time: 3, band: 3, y: 32, x: 32)>

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
    <xarray.DataArray (time: 39, band: 3, y: 128, x: 128)>

    Create a data tree with Sentinel-2 L2A and Sentinel-1 RTC from Planetary Computer, and
    Sentinel-2 Cloud Probability from Google Earth Engine:

    >>> import cubo
    >>> cubo.create(
    ...     lat=50,
    ...     lon=10,
    ...     collection=["sentinel-2-l2a", "sentinel-1-rtc", "COPERNICUS/S2_CLOUD_PROBABILITY"],
    ...     bands={
    ...         "sentinel-2-l2a": ["B04", "B08"],
    ...         "sentinel-1-rtc": ["vv", "vh"],
    ...         "COPERNICUS/S2_CLOUD_PROBABILITY": "probability",
    ...     },
    ...     stac={
    ...         "https://planetarycomputer.microsoft.com/api/stac/v1": ["sentinel-2-l2a", "sentinel-1-rtc"],
    ...         "ee": "COPERNICUS/S2_CLOUD_PROBABILITY",
    ...     },
    ...     start_date="2021-06-01",
    ...     end_date="2021-06-10",
    ...     edge_size=64,
    ...     resolution=10,
    ...     datatree=True,
    ... )
    <xarray.DataTree>
    """
    if xy_coords not in ["topleft", "center"]:
        raise ValueError(
            f"'xy_coords' must be 'topleft' or 'center', got '{xy_coords}'."
        )

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

    # Get one request per collection, and the keyword arguments of each one
    requests = _parse_collections(collection, stac, bands, gee)
    collections = [request["collection"] for request in requests]

    stackstac_kws = _split_kw(stackstac_kw, collections)
    gee_kws = _split_kw(gee_kw, collections)
    search_kws = _split_search_kw(kwargs, collections)

    if any("xy_coords" in kw for kw in stackstac_kws):
        raise ValueError(
            "'xy_coords' cannot be passed in 'stackstac_kw', use the 'xy_coords' argument instead."
        )

    cubes = []

    for request, stackstac_kw, gee_kw, search_kw in zip(
        requests, stackstac_kws, gee_kws, search_kws
    ):

        if request["stac"] == "ee":
            cube, name = _gee_cube(
                request["collection"],
                request["bands"],
                bbox_utm,
                bbox_latlon,
                epsg,
                resolution,
                start_date,
                end_date,
                gee_kw,
            )
            stac_endpoint = (
                "https://earthengine-stac.storage.googleapis.com/catalog/catalog.json"
            )
        else:
            cube = _stac_cube(
                request["collection"],
                request["stac"],
                request["bands"],
                bbox_utm,
                bbox_latlon,
                epsg,
                resolution,
                start_date,
                end_date,
                stackstac_kw,
                search_kw,
            )
            name = request["collection"]
            stac_endpoint = request["stac"]

        # Rounded edge size
        rounded_edge_size = cube.x.shape[0]

        # New attributes
        cube.attrs = dict(
            collection=name,
            stac=stac_endpoint,
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

        # The coordinates are pixel centers here
        cube = cube.assign_coords(
            {
                "cubo:distance_from_center": (
                    ["y", "x"],
                    _compute_distance_to_center(cube),
                )
            }
        )

        # Move the coordinates to the top-left corner of the pixels
        if xy_coords == "topleft":
            cube = cube.assign_coords(
                x=cube.x - resolution / 2, y=cube.y + resolution / 2
            )

        # New name
        cube.name = name

        cubes.append(cube)

    if datatree:
        return _to_datatree(cubes, dataset)

    if len(cubes) == 1:
        cube = cubes[0]
    else:
        cube = _merge(cubes)

    return _to_dataset(cube) if dataset else cube


def _stac_cube(
    collection: str,
    stac: str,
    bands: Optional[List[str]],
    bbox_utm: dict,
    bbox_latlon: dict,
    epsg: int,
    resolution: Union[float, int],
    start_date: str,
    end_date: str,
    stackstac_kw: dict,
    search_kw: dict,
) -> xr.DataArray:
    """Creates a data cube from a STAC Catalogue with the coordinates at the center of the pixels."""
    # Convert UTM Bbox to a Feature
    bbox_utm = rasterio.features.bounds(bbox_utm)

    # Open the Catalogue
    CATALOG = pystac_client.Client.open(stac)

    # Do a search
    SEARCH = CATALOG.search(
        intersects=bbox_latlon,
        datetime=f"{start_date}/{end_date}",
        collections=[collection],
        **search_kw,
    )

    # Get all items and sign if using Planetary Computer
    items = SEARCH.item_collection()

    if len(items) == 0:
        raise ValueError(
            f'No items found for collection "{collection}" in "{stac}" with the requested '
            "location, dates and search arguments."
        )

    if stac == "https://planetarycomputer.microsoft.com/api/stac/v1":
        items = pc.sign(items)

    # Create the cube
    cube = stackstac.stack(
        items,
        assets=bands,
        resolution=resolution,
        bounds=bbox_utm,
        epsg=epsg,
        xy_coords="center",
        **stackstac_kw,
    )

    # Delete attributes
    attributes = ["spec", "crs", "transform", "resolution"]

    for attribute in attributes:
        if attribute in cube.attrs:
            del cube.attrs[attribute]

    return cube


def _gee_cube(
    collection: Any,
    bands: Optional[List[str]],
    bbox_utm: dict,
    bbox_latlon: dict,
    epsg: int,
    resolution: Union[float, int],
    start_date: str,
    end_date: str,
    gee_kw: dict,
) -> tuple:
    """Creates a data cube from Google Earth Engine with the coordinates at the center of the pixels.

    Returns the cube and the name of the collection.
    """
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
        name = collection
        collection = ee.ImageCollection(collection)
    else:
        name = collection.get("system:id").getInfo()

    # Do the filtering: Bounds, time, and bands (the end date is included as in STAC)
    collection = collection.filterBounds(BBox).filterDate(
        start_date, _gee_end_date(end_date)
    )

    if bands is not None:
        collection = collection.select(bands)

    # Get BBox values in UTM to define the pixel grid
    W = bbox_utm["coordinates"][0][0][0]
    E = bbox_utm["coordinates"][0][2][0]
    N = bbox_utm["coordinates"][0][2][1]
    S = bbox_utm["coordinates"][0][0][1]

    # Add xee arguments, chunks can be overridden
    gee_kw = {"chunks": dict(), **gee_kw}

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

    return cube, name


def _merge(cubes: List[xr.DataArray]) -> xr.DataArray:
    """Merges cubes from multiple collections along the band dimension.

    The time dimension is the union of the timestamps of all cubes.
    """
    # Bands must be unique to be merged
    bands = [band for cube in cubes for band in cube.band.values]
    duplicated = sorted({band for band in bands if bands.count(band) > 1})

    if duplicated:
        raise ValueError(
            f"Bands {duplicated} are in more than one collection and cannot be merged, "
            "use 'datatree=True' instead."
        )

    # Metadata coordinates of each collection cannot be merged
    distance = (["y", "x"], cubes[0]["cubo:distance_from_center"].values)

    cubes = [
        cube.reset_coords(drop=True).assign_coords(
            collection=("band", [cube.name] * cube.band.shape[0])
        )
        for cube in cubes
    ]

    # Timestamps can be duplicated within a collection (e.g., overlapping tiles), so the
    # k-th occurrence of a timestamp is matched with the k-th occurrence in other collections
    keys = [
        list(
            zip(
                cube.time.values,
                pd.Series(cube.time.values).groupby(cube.time.values).cumcount(),
            )
        )
        for cube in cubes
    ]
    union = sorted(set(key for cube_keys in keys for key in cube_keys))
    positions = {key: i for i, key in enumerate(union)}

    # Use the positions as a unique time index for the outer join
    cubes = [
        cube.assign_coords(time=[positions[key] for key in cube_keys])
        for cube, cube_keys in zip(cubes, keys)
    ]

    cube = xr.concat(cubes, dim="band", join="outer", combine_attrs="drop")
    cube = cube.assign_coords(
        {
            "time": [key[0] for key in union],
            "cubo:distance_from_center": distance,
        }
    )

    # Attributes of all collections
    cube.attrs = dict(
        cubes[0].attrs,
        collection=[c.attrs["collection"] for c in cubes],
        stac=[c.attrs["stac"] for c in cubes],
    )
    cube.name = None

    return cube


def _to_dataset(cube: xr.DataArray) -> xr.Dataset:
    """Converts a cube to a dataset with one variable per band."""
    ds = cube.to_dataset("band")
    ds.attrs = cube.attrs

    # Keep the collection of each band when merging multiple collections
    if "collection" in cube.coords:
        ds = ds.drop_vars("collection", errors="ignore")
        for band, collection in zip(cube.band.values, cube.collection.values):
            ds[band].attrs["collection"] = str(collection)

    return ds


def _to_datatree(cubes: List[xr.DataArray], dataset: bool) -> xr.DataTree:
    """Creates a data tree with one node per collection and shared spatial coordinates."""
    # The root contains the shared spatial coordinates and attributes
    attrs = {
        key: value
        for key, value in cubes[0].attrs.items()
        if key not in ["collection", "stac"]
    }

    # Only values are used, so metadata coordinates of a collection are not inherited by others
    root = xr.Dataset(
        coords={
            "y": cubes[0].y.values,
            "x": cubes[0].x.values,
            "cubo:distance_from_center": (
                ["y", "x"],
                cubes[0]["cubo:distance_from_center"].values,
            ),
        },
        attrs=attrs,
    )

    nodes = {"/": root}

    for cube in cubes:
        node = cube.drop_vars("cubo:distance_from_center")
        node = _to_dataset(node) if dataset else node.to_dataset(name="data")
        node.attrs = cube.attrs
        nodes[cube.name] = node

    return xr.DataTree.from_dict(nodes)
