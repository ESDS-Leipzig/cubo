from typing import Any, List, Union

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

    return (bbox_utm, bbox_latlon, utm_coords, f"EPSG:{epsg}")
