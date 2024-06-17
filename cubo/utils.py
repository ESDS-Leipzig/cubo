from typing import Any, List, Union

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


def _get_epsg(
    lat: Union[float, int],
    lon: Union[float, int],
    edge_size: Union[float, int],
    resolution: Union[float, int],
    e7g: bool,
) -> int:
    """Gets the EPSG code of the UTM Zone or the Equi7Grid Zone.

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
    e7g : bool
        Whether to use Equi7Grid instead of UTM.

    Returns
    -------
    int
        EPSG code.
    """
    if e7g:

        # Try to import ee, otherwise raise an ImportError
        try:
            import equi7grid_lite
        except ImportError:
            raise ImportError(
                '"equi7grid-lite" could not be loaded. Please install it, or install "cubo" using "pip install cubo[e7g]"'
            )
        
        # EPSG list for Equi7Grid
        e7g_epsgs = dict(
            AN=27702, NA=27705, OC=27706, SA=27707, AF=27701, EU=27704, AS=27703
        )
        
        # Get the EPSG from latlon
        grid_system = equi7grid_lite.Equi7Grid(min_grid_size = edge_size * resolution)
        grid_polygon = grid_system.lonlat2grid(lon, lat)
        epsg = e7g_epsgs[grid_polygon.zone.values[0]]

    else:
        
        # Get the UTM EPSG from latlon
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(lon, lat, lon, lat),
        )
    
        # Save the CRS
        epsg = utm_crs_list[0].code

    return int(epsg)


def _central_pixel_bbox(
    lat: Union[float, int],
    lon: Union[float, int],
    edge_size: Union[float, int],
    resolution: Union[float, int],
    e7g: bool,
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
    e7g : bool
        Whether to use Equi7Grid instead of UTM.

    Returns
    -------
    tuple
        BBox in Equi7Grid coordinates, BBox in latlon, and EPSG.
    """
    # Get EPSG
    epsg = _get_epsg(lat, lon, edge_size, resolution, e7g)

    # Initialize a transformer to UTM
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    # Initialize a transformer from UTM to latlon
    inverse_transformer = Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
    )

    # Transform latlon to UTM
    proj_coords = transformer.transform(lon, lat)

    # Round the coordinates
    proj_coords_round = [round(coord / resolution) * resolution for coord in proj_coords]

    # Buffer size
    buffer = round(edge_size * resolution / 2)

    # Create BBox coordinates according to the edge size
    E = proj_coords_round[0] + buffer
    W = proj_coords_round[0] - buffer
    N = proj_coords_round[1] + buffer
    S = proj_coords_round[1] - buffer

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
    bbox_proj = {
        "type": "Polygon",
        "coordinates": [polygon],
    }

    # Create latlon BBox
    bbox_latlon = {
        "type": "Polygon",
        "coordinates": [polygon_latlon],
    }

    return (bbox_proj, bbox_latlon, proj_coords, epsg)


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
    distance_to_center = np.linalg.norm((coordinates) - np.array([x, y]), axis=0).T

    return distance_to_center
