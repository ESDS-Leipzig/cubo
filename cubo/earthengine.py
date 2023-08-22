from typing import List, Optional, Union
from datetime import datetime

import concurrent
import xarray as xr
import numpy as np
import ee

from .utils import (
    _ee_get_patch, _ee_fix_coordinates, 
    _ee_get_projection_metadata, _ee_fix_date
)


def create(
    lat: Union[float, int],
    lon: Union[float, int],
    collection: str,
    start_date: str,
    end_date: str,
    bands: Optional[Union[str, List[str]]] = None,
    edge_size: Union[float, int] = 128.0,
    max_workers: Union[float, int] = 200
) -> xr.DataArray:
    """Creates a data cube from a GEE Data Catalog as a :code:`xr.DataArray` object.

    The coordinates used here is aligned to the EE ImageCollection geotransform.

    Parameters
    ----------
    lat : float | int
        Latitude of the central pixel of the data cube.
    lon : float | int
        Longitude of the central pixel of the data cube.
    collection : str
        Name of the collection in the GEE Data Catalog.
    start_date : str
        Start date of the data cube in YYYY-MM-DD format.
    end_date : str
        End date of the data cube in YYYY-MM-DD format.
    bands : str | List[str], default = None
        Name of the band(s) from the collection to use.
    edge_size : float | int, default = 128
        Size of the edge of the cube in pixels. All edges share the same size.

        .. warning::
           If :code:`edge_size` is not a multiple of 2. Cubo will consider 1 pixel 
           more on the left and top edges.

    Returns
    -------
    xr.DataArray
        Data Cube.


    Examples
    --------
    Create a Sentinel-2 L1C data cube with an edge size of 64 px from Planetary Computer:

    >>> import cubo
    >>> cubo.create(
    ...     lat=50,
    ...     lon=10,
    ...     collection="COPERNICUS/S2",
    ...     bands=["B4","B3","B2"],
    ...     start_date="2021-06-01",
    ...     end_date="2021-06-10",
    ...     edge_size=64
    ... )
    <xarray.DataArray (time: 3, band: 3, x: 32, y: 32)>
    """
    
    # Create a ee.Geometry.Point
    ee_point = ee.Geometry.Point(lon, lat)
    
    # Obtain the projection (CRS and geotransform) parameters for the mini-cube
    projection_data = _ee_get_projection_metadata(collection, ee_point, start_date, end_date, bands)
    resolution_x = projection_data["transform"][0]
    resolution_y = projection_data["transform"][4]
    
    # Initialize a transformer to UTM
    epsg = projection_data["crs"]

    # Align the coordinates to the collection geotransform
    n_utm_x, n_utm_y  = _ee_fix_coordinates(projection_data, lon, lat)
    
    # Create BBox coordinates according to the edge size
    if edge_size % 2 != 0:
        utm_x = n_utm_x - (int(edge_size / 2) + 1) * resolution_x
        utm_y = n_utm_y - (int(edge_size / 2) + 1) * resolution_y
    else:
        utm_x = n_utm_x - int(edge_size / 2) * resolution_x
        utm_y = n_utm_y - int(edge_size / 2) * resolution_y
    
    # Create a EE Point (Center of the minicube)
    ee_geom = ee.Geometry.Point((n_utm_x, n_utm_y), proj=epsg)
    
    # Create a EE ImageCollection
    n_start_date, n_end_date = _ee_fix_date(start_date, end_date)
    
    ee_ic = (
        ee.ImageCollection(collection)
          .filterBounds(ee.Geometry(ee_geom))
          .filterDate(n_start_date, n_end_date)
          .select(bands)
    )
    
    # Obtain image IDs
    img_ids = ee_ic.aggregate_array('system:id').getInfo()
    img_dates = ee_ic.aggregate_array('system:time_start').getInfo()
    img_dates = [datetime.fromtimestamp(date / 1000) for date in img_dates]    
    zip_list = list(zip(img_ids, img_dates))
        
    # Define the executor
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    # Parallelize the patch download
    point = (utm_x, utm_y)
    resolution = (resolution_x, resolution_y)
    dimension = (edge_size, edge_size)
    future_to_image = {
        executor.submit(_ee_get_patch, point, resolution, dimension, img_id[0], bands, epsg):
            img_id for img_id in zip_list
    }
    
    arrays = ()
    time = []
    img_ids = []
    for future in concurrent.futures.as_completed(future_to_image):
        
        img_id, img_date = future_to_image[future]
        
        try:
            np_array = future.result()
            time += (img_date,)
            img_ids += (img_id,)
            arrays += (np_array,)
            
        except Exception as e:
            print(e)
            pass
    
    # Create the xarray coordinates
    range_y = np.arange(utm_y, utm_y + edge_size * resolution_y, resolution_y)[0:edge_size]
    range_x = np.arange(utm_x, utm_x + edge_size * resolution_x, resolution_x)[0:edge_size]
    
    # from a numpy create a xarray
    if bands is str:
        bands = [bands]
    
    array = xr.DataArray(
        data=np.array(arrays),
        dims=["time", "band", "y", "x"],
        coords= dict(
            time = time,
            band = bands,
            y = range_y,
            x = range_x
        ),
        attrs=dict(
            collection=collection,
            epsg=epsg,
            resolution=(resolution_x, resolution_y),
            edge_size=edge_size,
            central_lat=lat,
            central_lon=lon,
            central_y=n_utm_y + resolution_y / 2,
            central_x=utm_x + resolution_x / 2,
            time_coverage_start=start_date,
            time_coverage_end=end_date,
            gee_image_ids = img_ids
        )
    )
    
    return array.sortby("time")