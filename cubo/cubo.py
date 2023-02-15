from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio.features
import stackstac
import xarray as xr

from .utils import _central_pixel_bbox


def create(
    lat: Union[float, int],
    lon: Union[float, int],
    collection: str,    
    start_date: str,
    end_date: str,
    bands: Optional[Union[str, List[str]]] = None,
    edge_size: Union[float, int] = 128.0,
    resolution: Union[float, int] = 10.0,
    stac: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
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
        Size of the edge of the cube in pixels. All edges share the same size.

        .. warning::
           If :code:`edge_size` is not a multiple of 2, it will be rounded.

    resolution : float | int, default = 10
        Pixel size in meters.
    stac : str, default = 'https://planetarycomputer.microsoft.com/api/stac/v1'
        Endpoint of the STAC Catalogue to use.
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
    """
    # Get the BBox and EPSG
    bbox_utm, bbox_latlon, utm_coords, epsg = _central_pixel_bbox(
        lat, lon, edge_size, resolution
    )

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
    items = SEARCH.get_all_items()

    if stac == "https://planetarycomputer.microsoft.com/api/stac/v1":
        items = pc.sign(items)

    # Put the bands into list if not a list already
    if not isinstance(bands, list) and bands is not None:
        bands = [bands]

    # Create the cube
    cube = stackstac.stack(
        items,
        assets=bands,
        resolution=resolution,
        bounds=bbox_utm,
        epsg=epsg,
    )

    # Delete attributes
    attributes = ["spec", "crs", "transform", "resolution"]

    for attribute in attributes:
        if attribute in cube.attrs:
            del cube.attrs[attribute]

    # New attributes
    cube.attrs = dict(
        collection=collection,
        stac=stac,
        epsg=epsg,
        resolution=resolution,
        edge_size=edge_size,
        central_lat=lat,
        central_lon=lon,
        central_y=utm_coords[1],
        central_x=utm_coords[0],
        time_coverage_start=start_date,
        time_coverage_end=end_date,
    )

    # New name
    cube.name = collection

    return cube
