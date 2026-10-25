from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


def _central_pixel_bbox(
    lat: Union[float, int],
    lon: Union[float, int],
    edge_size: Union[float, int],
    resolution: Union[float, int],
) -> tuple:
    """Creates a Bounding Box (BBox) given a pair of coordinates and a buffer distance.

    Parameters
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.
    edge_size : float
        Buffer distance in meters.
    resolution : int | float
        Spatial resolution to use.
    latlng : bool, default = True
        Whether to return the BBox as geographic coordinates.

    Returns
    -------
    tuple
        BBox in UTM coordinates, BBox in latlon, and EPSG.
    """
    # Get the UTM EPSG from latlon
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(lon, lat, lon, lat),
    )

    # Save the CRS
    epsg = utm_crs_list[0].code
    utm_crs = CRS.from_epsg(epsg)

    # Initialize a transformer to UTM
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    # Initialize a transformer from UTM to latlon
    inverse_transformer = Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
    )

    # Transform latlon to UTM
    utm_coords = transformer.transform(lon, lat)

    # Round the coordinates
    utm_coords_round = [round(coord / resolution) * resolution for coord in utm_coords]

    # Buffer size
    buffer = round(edge_size * resolution / 2)

    # Create BBox coordinates according to the edge size
    E = utm_coords_round[0] + buffer
    W = utm_coords_round[0] - buffer
    N = utm_coords_round[1] + buffer
    S = utm_coords_round[1] - buffer

    # Create polygon from BBox coordinates
    polygon = [
        [W, S],
        [E, S],
        [E, N],
        [W, N],
        [W, S],
    ]

    # Transform vertices of polygon to latlon
    polygon_latlon = [list(inverse_transformer.transform(x[0], x[1])) for x in polygon]

    # Create UTM BBox
    bbox_utm = {
        "type": "Polygon",
        "coordinates": [polygon],
    }

    # Create latlon BBox
    bbox_latlon = {
        "type": "Polygon",
        "coordinates": [polygon_latlon],
    }

    return (bbox_utm, bbox_latlon, utm_coords, int(epsg))


def _compute_distance_to_center(da: xr.DataArray) -> np.ndarray:
    """Computes the distance from each pixel to the specified center of the data cube.

    Parameters
    ----------
    da : xr.DataArray
        Data cube to compute the distance from.

    Returns
    -------
    np.ndarray
        Distance from each pixel to the specified center.
    """
    # Create meshgrid of coordinates
    coordinates = np.meshgrid(da.x, da.y)

    # Create meshgrid using just the value of the center coordinates
    x = (coordinates[0] ** 0) * da.attrs["central_x"]
    y = (coordinates[1] ** 0) * da.attrs["central_y"]

    # Compute the distance, the meshgrid is already (y, x)
    distance_to_center = np.linalg.norm((coordinates) - np.array([x, y]), axis=0)

    return distance_to_center


def _parse_collections(
    collection: Any,
    stac: Union[str, dict],
    bands: Optional[Union[str, List[str], dict]],
    gee: bool,
) -> List[dict]:
    """Harmonizes the collections, STAC endpoints and bands to one request per collection.

    Parameters
    ----------
    collection : str | List[str] | ee.ImageCollection
        Collection(s) to use.
    stac : str | dict
        Endpoint of the STAC Catalogue, or a dictionary with endpoints as keys and the
        collection(s) of each endpoint as values. The 'ee' key is used for Google Earth Engine.
    bands : str | List[str] | dict
        Band(s) to use, or a dictionary with collections as keys and the band(s) of each
        collection as values.
    gee : bool
        Whether to use Google Earth Engine for all collections. This ignores the 'stac' argument.

    Returns
    -------
    List[dict]
        Requests with the collection, STAC endpoint and bands of each collection.
    """
    # Put the collections into a list if not a list already
    collections = collection if isinstance(collection, list) else [collection]

    if len(collections) == 0:
        raise ValueError("At least one collection must be provided.")

    # Collections must be strings to be matched with the 'stac' and 'bands' dictionaries
    named = all(isinstance(c, str) for c in collections)

    if not named and (
        len(collections) > 1 or isinstance(stac, dict) or isinstance(bands, dict)
    ):
        raise TypeError(
            "Collections must be strings when requesting multiple collections, or when "
            "'stac' or 'bands' are dictionaries."
        )

    if named and len(set(collections)) != len(collections):
        raise ValueError(f"Duplicated collections are not allowed: {collections}.")

    # Using gee=True is equivalent to stac='ee'
    if gee:
        stac = "ee"

    # Get the STAC endpoint of each collection
    if isinstance(stac, dict):
        endpoints = dict()
        for endpoint, values in stac.items():
            values = values if isinstance(values, list) else [values]
            for value in values:
                if value in endpoints:
                    raise ValueError(
                        f'Collection "{value}" is assigned to more than one STAC endpoint '
                        f'("{endpoints[value]}" and "{endpoint}"). Requesting the same '
                        "collection from different catalogues is not supported yet."
                    )
                endpoints[value] = endpoint

        unknown = [c for c in endpoints if c not in collections]
        missing = [c for c in collections if c not in endpoints]

        if unknown:
            raise ValueError(
                f"Collections in 'stac' are not in 'collection': {unknown}."
            )
        if missing:
            raise ValueError(
                f"Collections in 'collection' have no STAC endpoint in 'stac': {missing}."
            )

        endpoints = [endpoints[c] for c in collections]
    else:
        endpoints = [stac for c in collections]

    # Get the bands of each collection
    if isinstance(bands, dict):
        unknown = [c for c in bands if c not in collections]
        if unknown:
            raise ValueError(
                f"Collections in 'bands' are not in 'collection': {unknown}."
            )
        bands = [bands.get(c) for c in collections]
    elif len(collections) > 1 and bands is not None:
        raise TypeError(
            "When requesting multiple collections, 'bands' must be a dictionary with "
            "collections as keys and the band(s) of each collection as values."
        )
    else:
        bands = [bands]

    # Put the bands into list if not a list already
    bands = [[b] if isinstance(b, str) else b for b in bands]

    return [
        dict(collection=c, stac=s, bands=b)
        for c, s, b in zip(collections, endpoints, bands)
    ]


def _split_kw(kw: Optional[dict], collections: List[Any]) -> List[dict]:
    """Splits global or per-collection keyword arguments into one dictionary per collection.

    The keyword arguments are per collection if all their keys are requested collections.

    Parameters
    ----------
    kw : dict
        Global keyword arguments, or a dictionary with collections as keys and the keyword
        arguments of each collection as values.
    collections : List[Any]
        Requested collections.

    Returns
    -------
    List[dict]
        Keyword arguments of each collection.
    """
    kw = kw or dict()
    names = [c for c in collections if isinstance(c, str)]

    if kw and all(key in names for key in kw):
        return [dict(kw.get(c) or dict()) for c in collections]

    return [dict(kw) for c in collections]


def _split_search_kw(kw: dict, collections: List[Any]) -> List[dict]:
    """Splits global or per-collection search keyword arguments into one dictionary per collection.

    Each search keyword argument is per collection if its value is a dictionary whose keys
    are all requested collections.

    Parameters
    ----------
    kw : dict
        Search keyword arguments.
    collections : List[Any]
        Requested collections.

    Returns
    -------
    List[dict]
        Search keyword arguments of each collection.
    """
    names = [c for c in collections if isinstance(c, str)]
    split = [dict() for c in collections]

    for key, value in kw.items():
        per_collection = (
            isinstance(value, dict) and value and all(k in names for k in value)
        )
        for i, c in enumerate(collections):
            if not per_collection:
                split[i][key] = value
            elif c in value:
                split[i][key] = value[c]

    return split


def _gee_end_date(end_date: str) -> str:
    """Converts an end date to the exclusive end date of Google Earth Engine.

    STAC searches include the end date, while Google Earth Engine excludes it. This returns the
    next day for dates (YYYY-MM-DD), or the next millisecond for datetimes, so the end date is
    included in Google Earth Engine too.

    Parameters
    ----------
    end_date : str
        End date.

    Returns
    -------
    str
        Exclusive end date for Google Earth Engine.
    """
    timestamp = pd.Timestamp(end_date)

    if len(str(end_date)) == 10:
        timestamp = timestamp + pd.Timedelta(days=1)
    else:
        timestamp = timestamp + pd.Timedelta(milliseconds=1)

    return timestamp.isoformat()
