Changelog
=========

v2026.10.0
----------

- Fix: :code:`cubo:distance_from_center` was transposed, now it matches the :code:`(y, x)` pixel coordinates.
- Added :code:`pyproj` and :code:`rasterio` as explicit dependencies.
- Synced :code:`requirements.txt` with :code:`pyproject.toml`.
- Migrated to :code:`xee>=0.1.0` (breaking changes in :code:`xee`): the GEE pixel grid is now defined explicitly and matches the :code:`stackstac` grid.
- Added the :code:`gee_kw` argument to :code:`cubo.create()`.
- Added support for multiple collections in :code:`cubo.create()`: :code:`collection` can be a list,
    :code:`bands` can be a dictionary with the bands of each collection, and :code:`stac` can be a dictionary
    with the collections of each endpoint (use the :code:`"ee"` key for Google Earth Engine). All collections share the same pixel grid.
- Added the :code:`datatree` argument to :code:`cubo.create()` to return a :code:`xr.DataTree` with one node per collection.
- Added the :code:`dataset` argument to :code:`cubo.create()` to return a :code:`xr.Dataset` with one variable per band.
- Added the :code:`xy_coords` argument to :code:`cubo.create()`: :code:`"topleft"` (default) or :code:`"center"`, for both STAC and Google Earth Engine.
- :code:`stackstac_kw`, :code:`gee_kw` and the search keyword arguments can be set per collection.
- Breaking: :code:`end_date` is now included for Google Earth Engine too (as in STAC Catalogues).
- Breaking: :code:`cubo:distance_from_center` is now always computed from the center of the pixels.
- Breaking: :code:`xy_coords` cannot be passed in :code:`stackstac_kw` anymore, use the :code:`xy_coords` argument instead.
- Clearer error when no items are found for a STAC collection.
- Pinned: :code:`xarray>=2024.10.0` (required for :code:`xr.DataTree`).
- Added support for Python 3.11, 3.12, and 3.13.
- Dropped support for Python 3.10.
- Fixed the default value of :code:`gee` in the :code:`cubo.create()` docstring.
- Extended the tests and added offline tests for the utilities.

v2026.2.0
---------

- Migrated from :code:`setup.py` to :code:`pyproject.toml`.
- :code:`numpy` was relaxed from :code:`numpy<2.0.0`.
- Pinned: :code:`stackstac>=0.5.1`.

v2024.8.0
---------

- Fix: Avoid Google Earth Engine initialization within :code:`cubo`.

v2024.6.0
---------

- Pinned: :code:`numpy<2.0.0` as :code:`stackstac` breaks with :code:`numpy>=2.0.0`.

v2024.1.1
---------

- Added the :code:`units` argument to :code:`cubo.create()`.
- Added support for :code:`scipy.constants` units.

v2024.1.0
---------

- Added support for Google Earth Engine.
- Added the :code:`gee` argument to :code:`cubo.create()`.
- Added support for :code:`stackstac` keyword arguments.
- Added the :code:`stackstac_kw` argument to :code:`cubo.create()`.

v2023.12.0
---------

- Added preservation via Zenodo.

v2023.7.2
---------

- Added :code:`cubo:distance_from_center` coordinate.

v2023.7.1
---------

- Replaced :code:`get_all_items` by :code:`item_collection`.

v2023.7.0
---------

- Fixed the required datatype of the EPSG code for stackstac `(#4) <https://github.com/ESDS-Leipzig/cubo/issues/4>`_.
- Pinned latest versions: :code:`dask>=2023.7.0`, :code:`pandas>=2.0.3`, 
    :code:`planetary_computer>=1.0.0`, :code:`pystac_client>=0.7.2`, :code:`stackstac>=0.4.4`, and :code:`xarray>=2023.6.0`.

v0.1.0
------

- First release of :code:`cubo`!