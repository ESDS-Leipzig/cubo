from typing import Union, List
from datetime import timedelta, datetime
import numpy as np
import xarray as xr
from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from google.api_core import retry
import io
import ee
import warnings
import requests


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

    # Compute the distance, transposed, so y is first
    distance_to_center = np.linalg.norm((coordinates) - np.array([x, y]), axis=0).T

    return distance_to_center


@retry.Retry()
def _ee_get_patch(
    point: tuple,
    resolution: tuple,
    dimension: tuple,
    asset_id: str,
    band: Union[str, int],
    crs: str
) -> np.ndarray:
    """Get a patch of pixels from an asset, centered on the coords.

    Args:
        geometry (dict): _description_
        asset_id (str): _description_
        band (Union[str, int]): _description_

    Returns:
        np.ndarray: _description_
    """
    # Get the bands
    if isinstance(band, str):
        band = [band]
    
    # Create the file request
    request = {
        'fileFormat': 'NPY',
        'bandIds': band,
        'assetId': asset_id,
        'grid': {
        'dimensions': {
            'width': dimension[0],
            'height': dimension[1]
        },
        'affineTransform': {
            'scaleX': resolution[0],
            'shearX': 0,
            'translateX': point[0],
            'shearY': 0,
            'scaleY': resolution[1],
            'translateY': point[1]
        },
        'crsCode': crs,
    }
    }
    
    # Get the data
    minicube_layer = np.load(io.BytesIO(ee.data.getPixels(request)))
    minicube_layer = np.stack([minicube_layer[x] for x in band]) # Sort the bands
    
    return minicube_layer


def _ee_get_coordinates(
    projection_data: dict,
    lon: Union[float, int],
    lat: Union[float, int]
) -> tuple:
    """ Fix the central coordinates of the minicube according to 
        the EE ImageCollection geotransform.

    Args:
        projection_data (dict): The projection parameters of the minicube.
        lon (Union[float, int]): The longitude.
        lat (Union[float, int]): The latitude.

    Returns:
        tuple: The new coordinates.
    """
    # Get the CRS
    crs = projection_data["crs"]
    
    # get the CRS in WKT2 format
    crs_wkt = _ee_utils_get_crs(crs) # Useful for ESRI and SR-ORG (e.g. MODIS images)

    # Local coordinates
    transformer_local = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)
    local_coords = transformer_local.transform(lon, lat)

    return {"geo": (lon, lat), "local": local_coords}


def _ee_fix_coordinates(
    projection_data: dict,
    lon: Union[float, int],
    lat: Union[float, int]
) -> tuple:
    """ Fix the central coordinates of the minicube according to 
        the EE ImageCollection geotransform.

    Args:
        projection_data (dict): The projection parameters of the minicube.
        lon (Union[float, int]): The longitude.
        lat (Union[float, int]): The latitude.

    Returns:
        tuple: The new coordinates.
    """
    
    # Get the CRS
    crs = projection_data["crs"]
    
    # Get the coordinates of the upper left corner
    ulx = projection_data["transform"][2]
    uly = projection_data["transform"][5]
    
    # Get the scale of the minicube
    scale_x = projection_data["transform"][0]
    scale_y = projection_data["transform"][4]
    
    # From WGS84 to UTM     
    crs_wkt = _ee_utils_get_crs(crs) # Useful for ESRI and SR-ORG (e.g. MODIS images)
    transformer_local = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)
    local_coords = transformer_local.transform(lon, lat)
    
    # Fix the coordinates
    display_x = round((local_coords[0] - ulx) / scale_x)
    display_y = round((local_coords[1] - uly) / scale_y)

    # New local coordinates
    new_x = ulx + display_x * scale_x
    new_y = uly + display_y * scale_y
    
    # New geographic coordinates
    transformer_geo = Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)
    new_lon, new_lat = transformer_geo.transform(new_x, new_y)
    
    return {"geo": (new_lon, new_lat), "local": (new_x, new_y)}


# collection, ee_geom, start_date, end_date, bands = collection, ee_point, start_date, end_date, bands
def _ee_get_projection_metadata(
    collection: str,
    ee_geom: ee.Geometry,
    start_date: str,
    end_date: str,
    bands: Union[str, List[str]] = None
) ->  dict:
    """ Get the projection parameters for the minicube

    Args:
        collection (str): The EE ImageCollection.
        ee_geom (ee.Geometry): A EE Geometry to filter the ImageCollection.
        start_date (str): The start date.
        end_date (str): The end date.
        bands (Optional[str, List[str]], optional): The bands to use. Defaults to None.

    Returns:
        dict: The projection parameters.
    """
    
    # Get the transform parameters of the first image
    ee_img = (
          collection
          .filterBounds(ee_geom)
          .filterDate(start_date, end_date)
          .first()
          .select(bands)
    )
    
    try:
        info = ee_img.projection().getInfo()
    except Exception as e:
        warnings.warn(
            "The bands of the specified image contains different projections. Using the first band."
        )
        try:
            info = ee_img.select(0).projection().getInfo()
        except Exception as e:
            raise ValueError(
                "An error occurred while trying to get image metadata. "
            )
    return info
    


def _ee_fix_date(start_date: str, end_date: str) -> tuple:
    """Fixes the start and end dates to be used in the GEE API.

    Args:
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.

    Returns:
        tuple: The fixed start and end dates in YYYY-MM-DD format.
    """
    delta = timedelta(days=1)
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    start_date = start_date.strftime('%Y-%m-%d')
    
    end_date = datetime.strptime(end_date, '%Y-%m-%d') + delta
    end_date = end_date.strftime('%Y-%m-%d')
    
    return start_date, end_date


def _ee_utils_get_crs(code: str) -> str:
    """ Get the EPSG code from a EPSG, ESRI or SR-ORG code

    Args:
        code (str): The projection code.

    Returns:
        str: The EPSG code.
    """
    codetype = code.split(":")[0].lower()
    if codetype == "epsg":
        return code
    return _get_crs_web(code)


def _get_crs_web(code: str) -> str:
    """ Convert EPSG, ESRI or SR-ORG code into a OGC WKT
    
    Args:
        code (str): The projection code.

    Returns:
        str: A string representing the same projection but in WKT2 format.
    """
    
    # Get the code type and the code
    codetype, ee_code = code.split(":")
    codetype = codetype.lower()
    
    if codetype == "epsg":
        link = f"https://epsg.io/{ee_code}.wkt"
        crs_wkt = requests.get(link).text
    else:
        link = f"https://spatialreference.org/ref/{codetype}/{ee_code}/ogcwkt/"
        try:
            crs_wkt = requests.get(link).text
        except requests.exceptions.RequestException:
            print("spatialreference.org is down using web.archive.org ...")
            link = f"https://web.archive.org/web/https://spatialreference.org/ref/{codetype}/{ee_code}/ogcwkt/"
            crs_wkt = requests.get(link).text            
    return crs_wkt