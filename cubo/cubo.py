from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .utils import *


def create(    
    lat: Union[float, int],
    lon: Union[float, int],
    collection: str,
    bands: Union[str, List[str]],
    start_date: str,
    end_date: str,
    edge_size: Union[float, int] = 128.0,
    resolution: Union[float, int] = 10.0,
    stac: str = 'https://planetarycomputer.microsoft.com/api/stac/v1',    
) -> xr.Dataset:
    """Creates a data cube from a STAC Catalogue as a :code:`xr.Dataset` object.

    Parameters
    ----------
    lat : float | int
        Latitude of the central pixel of the data cube.
    lon : float | int
        Longitude of the central pixel of the data cube.
    collection : str
        Name of the collection in the STAC Catalogue.
    bands : str | List[str]
        Name of the band(s) from the collection to use.
    start_date : str
        Start date of the data cube in YYYY-MM-DD format.
    end_date : str
        End date of the data cube in YYYY-MM-DD format.
    edge_size : float | int, default = 128
        Size of the edge of the cube in pixels. All edges share the same size.
    resolution : float | int, default = 10
        Pixel size in meters.
    stac : str, default = 'https://planetarycomputer.microsoft.com/api/stac/v1'
        Endpoint of the STAC Catalogue to use.

    Returns
    -------
    xr.Dataset
        Data Cube.


    Examples
    --------
    Create a Sentinel-2 L2A data cube with an edge size of 64 px from Planetary Computer:

    >>> import cubo
    >>> cubo.create(
    ...     lat = 3.71,
    ...     lon = -76.43,
    ...     collection = "sentinel-2-l2a"
    ...     resolution = 10,
    ...     start_date = "2021-01-01",
    ...     end_date = "2021-02-01",
    ...     edge_size = 64
    ... )
    <xarray.DataArray (time: 12, x: 64, y: 64)>
    """

    result = 1

    return result