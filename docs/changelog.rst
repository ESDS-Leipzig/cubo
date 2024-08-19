Changelog
=========

v2024.8.0
---------

- Fix: Avoind Google Earth Engine initialization within :code:`cubo`.

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