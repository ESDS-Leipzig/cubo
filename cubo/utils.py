from typing import Any, List, Union

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


def _bbox_to_utm(
    bbox: tuple,
    resolution: Union[float, int],
) -> tuple:
    """Converts a bounding box in EPSG:4326 to UTM coordinates.

    Parameters
    ----------
    bbox : tuple
        Bounding box as (minx, miny, maxx, maxy) in EPSG:4326.
    resolution : int | float
        Spatial resolution to use.

    Returns
    -------
    tuple
        BBox in UTM coordinates, BBox in latlon, center UTM coords, and EPSG.
    """
    minx, miny, maxx, maxy = bbox
    
    # Calculate center for UTM zone detection
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2
    
    # Get the UTM EPSG from center
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            center_lon, center_lat, center_lon, center_lat
        ),
    )
    
    epsg = utm_crs_list[0].code
    
    # Initialize transformer to UTM
    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )
    
    # Transform bbox corners to UTM
    W, S = transformer.transform(minx, miny)
    E, N = transformer.transform(maxx, maxy)
    
    # Round to resolution - expand to cover fully
    W = np.floor(W / resolution) * resolution
    S = np.floor(S / resolution) * resolution
    E = np.ceil(E / resolution) * resolution
    N = np.ceil(N / resolution) * resolution
    
    # Create polygon from BBox coordinates
    polygon = [
        [W, S],
        [E, S],
        [E, N],
        [W, N],
        [W, S],
    ]
    
    # Create latlon polygon
    polygon_latlon = [
        [minx, miny],
        [maxx, miny],
        [maxx, maxy],
        [minx, maxy],
        [minx, miny],
    ]
    
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
    
    # Calculate center in UTM
    center_x = (W + E) / 2
    center_y = (S + N) / 2
    utm_coords = (center_x, center_y)
    
    return (bbox_utm, bbox_latlon, utm_coords, int(epsg))


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
    utm_coords_round = [
        round(coord / resolution) * resolution for coord in utm_coords
    ]

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
    polygon_latlon = [
        list(inverse_transformer.transform(x[0], x[1])) for x in polygon
    ]

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

    # Compute the distance, transposed, so y is first
    distance_to_center = np.linalg.norm(
        (coordinates) - np.array([x, y]), axis=0
    ).T

    return distance_to_center
