.. toctree::
   :maxdepth: 2
   :hidden:
  
   reference
   tutorials
   changelog

Cubo
====

.. raw:: html

   <p align="center">
   <a href="https://github.com/davemlz/cubo"><img src="https://github.com/davemlz/cubo/raw/main/docs/_static/logo.png" alt="cubo"></a>
   </p>
   <p align="center">
      <em>On-Demand Earth System Data Cubes (ESDCs) in Python</em>
   </p>
   <p align="center">
   <a href='https://pypi.python.org/pypi/cubo'>
      <img src='https://img.shields.io/pypi/v/cubo.svg' alt='PyPI' />
   </a>
   <a href='https://anaconda.org/conda-forge/cubo'>
      <img src='https://img.shields.io/conda/vn/conda-forge/cubo.svg' alt='conda-forge' />
   </a>
   <a href='https://cubo.readthedocs.io/en/latest/?badge=latest'>
      <img src='https://readthedocs.org/projects/cubo/badge/?version=latest' alt='Documentation Status' />
   </a>
   <a href='https://arxiv.org/abs/2404.13105'>
    <img src='https://img.shields.io/badge/arXiv-2404.13105-b31b1b.svg' alt='Documentation Status' />
   </a>
   <a href="https://github.com/davemlz/cubo/actions/workflows/tests.yml" target="_blank">
      <img src="https://github.com/davemlz/cubo/actions/workflows/tests.yml/badge.svg" alt="Tests">
   </a>
   <a href="https://opensource.org/licenses/MIT" target="_blank">
      <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
   </a>
   <a href="https://github.com/sponsors/davemlz" target="_blank">
      <img src="https://img.shields.io/badge/GitHub%20Sponsors-Donate-ff69b4.svg" alt="GitHub Sponsors">
   </a>
   <a href="https://www.buymeacoffee.com/davemlz" target="_blank">
      <img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-Donate-ff69b4.svg" alt="Buy me a coffee">
   </a>
   <a href="https://ko-fi.com/davemlz" target="_blank">
      <img src="https://img.shields.io/badge/kofi-Donate-ff69b4.svg" alt="Ko-fi">
   </a>
   <a href="https://twitter.com/dmlmont" target="_blank">
      <img src="https://img.shields.io/twitter/follow/dmlmont?style=social" alt="Twitter">
   </a>
   <a href="https://github.com/psf/black" target="_blank">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
   </a>
   <a href="https://pycqa.github.io/isort/" target="_blank">
      <img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="isort">
   </a>
   </p>

Overview
--------

SpatioTemporal Asset Catalogs (STAC) provide a standardized format that describes
geospatial information. Multiple platforms are using this standard to provide clients 
several datasets. Nice platforms such as Planetary Computer use this standard.
Additionally, Google Earth Engine (GEE) also provides a gigantic catalogue that users can 
harness for different tasks in Python.

`cubo` is a Python package that provides users of STAC and GEE an easy way to create 
On-demand Earth System Data Cubes (ESDCs). This is perfectly suitable for 
Deep Learning (DL) tasks. You can easily create a lot of ESDCs by just knowing a pair 
of coordinates and the edge size of the cube in pixels!

Check the simple usage of `cubo` with STAC here:

.. code-block:: python

   import cubo
   import xarray as xr

   da = cubo.create(
      lat=4.31, # Central latitude of the cube
      lon=-76.2, # Central longitude of the cube
      collection="sentinel-2-l2a", # Name of the STAC collection
      bands=["B02","B03","B04"], # Bands to retrieve
      start_date="2021-06-01", # Start date of the cube
      end_date="2021-06-10", # End date of the cube
      edge_size=64, # Edge size of the cube (px)
      resolution=10, # Pixel size of the cube (m)
   )

.. raw:: html

    <embed>
        <p align="center">
            <a href="https://github.com/davemlz/cubo"><img src="https://github.com/davemlz/cubo/raw/main/docs/_static/cubo_desc.png" alt="Cubo Description"></a>
        </p>
    </embed>

This chunk of code just created an :code:`xr.DataArray` object given a pair of 
coordinates, the edge size of the cube (in pixels), and additional information to get the 
data from STAC (Planetary Computer by default, but you can use another provider!). Note 
that you can also use the resolution you want (in meters) and the bands that you require.

Now check the simple usage of `cubo` with GEE here:

.. code-block:: python

   import cubo
   import xarray as xr

   da = cubo.create(
      lat=51.079225, # Central latitude of the cube
      lon=10.452173, # Central longitude of the cube
      collection="COPERNICUS/S2_SR_HARMONIZED", # Id of the GEE collection
      bands=["B2","B3","B4"], # Bands to retrieve
      start_date="2016-06-01", # Start date of the cube
      end_date="2017-07-01", # End date of the cube
      edge_size=128, # Edge size of the cube (px)
      resolution=10, # Pixel size of the cube (m)
      gee=True # Use GEE instead of STAC
   )

This chunk of code is very similar to the STAC-based cubo code. Note that the :code:`collection`
is now the ID of the GEE collection to use, and note that the :code:`gee` argument must be set to
:code:`True`.

How does it work?
-----------------

The thing is super easy and simple.

1. You have the coordinates of a point of interest. The cube will be created around these 
coordinates (i.e., these coordinates will be approximately the spatial center of the cube).
2. Internally, the coordinates are transformed to the projected UTM coordinates [x,y] 
in meters (i.e., local UTM CRS). They are rounded to the closest pair of coordinates 
that are divisible by the resolution you requested.
3. The edge size you provide is used to create a Bounding Box (BBox) for the cube in the 
local UTM CRS given the exact amount of pixels (Note that the edge size should be a 
multiple of 2, otherwise it will be rounded, usual edge sizes for ML are 64, 128, 256, 
512, etc.).
4. Additional information is used to retrieve the data from the STAC catalogue or from GEE: starts 
and end dates, name of the collection, endpoint of the catalogue (ignored for GEE), etc.
5. Then, by using :code:`stackstac` and :code:`pystac_client` the cube is retrieved as a 
:code:`xr.DataArray`. In the case of GEE, the cube is retrieved via :code:`xee`.
6. Success! That's what `cubo` is doing for you, and you just need to provide the 
coordinates, the edge size, and the additional info to get the cube.

Installation
------------

Install the latest version from PyPI:

.. code-block::

   pip install cubo


Install `cubo` with the required GEE dependencies from PyPI:

.. code-block::

   pip install cubo[ee]

Upgrade `cubo` by running:

.. code-block::

   pip install -U cubo


Install the latest version from conda-forge:

.. code-block::
   
   conda install -c conda-forge cubo


Install the latest dev version from GitHub by running:

.. code-block::

   pip install git+https://github.com/davemlz/cubo


Features
--------

Main function: `create()`
~~~~~~~~~~~~~~~~~~~~~~~~~

`cubo` is pretty straightforward, everything you need is in the `create()` function:

.. code-block:: python
   
   da = cubo.create(
      lat=4.31,
      lon=-76.2,
      collection="sentinel-2-l2a",
      bands=["B02","B03","B04"],
      start_date="2021-06-01",
      end_date="2021-06-10",
      edge_size=64,
      resolution=10,
   )

Using different units for `edge_size`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the units of `edge_size` are pixels. But you can modify this using the `units` argument:

.. code-block:: python

   da = cubo.create(
      lat=4.31,
      lon=-76.2,
      collection="sentinel-2-l2a",
      bands=["B02","B03","B04"],
      start_date="2021-06-01",
      end_date="2021-06-10",
      edge_size=1500,
      units="m",
      resolution=10,
   )

You can use "px" (pixels), "m" (meters), or any unit available in
 `scipy.constants <https://docs.scipy.org/doc/scipy/reference/constants.html#units>`_.

.. code-block:: python

   da = cubo.create(
      lat=4.31,
      lon=-76.2,
      collection="sentinel-2-l2a",
      bands=["B02","B03","B04"],
      start_date="2021-06-01",
      end_date="2021-06-10",
      edge_size=1.5,
      units="kilo",
      resolution=10,
   )

Using another endpoint
~~~~~~~~~~~~~~~~~~~~~~

By default, `cubo` uses Planetary Computer. But you can use another STAC provider endpoint if you want:

.. code-block:: python

   da = cubo.create(
      lat=4.31,
      lon=-76.2,
      collection="sentinel-s2-l2a-cogs",
      bands=["B05","B06","B07"],
      start_date="2020-01-01",
      end_date="2020-06-01",
      edge_size=128,
      resolution=20,
      stac="https://earth-search.aws.element84.com/v0"
   )

Keywords for searching data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can pass `kwargs` to `pystac_client.Client.search()` if required:

.. code-block:: python
   
   da = cubo.create(
      lat=4.31,
      lon=-76.2,
      collection="sentinel-2-l2a",
      bands=["B02","B03","B04"],
      start_date="2021-01-01",
      end_date="2021-06-10",
      edge_size=64,
      resolution=10,
      query={"eo:cloud_cover": {"lt": 10}} # kwarg to pass
   )

License
-------

The project is licensed under the MIT license.

Logo Attribution
----------------

The logo and images were created using `dice icons created by Freepik - Flaticon <https://www.flaticon.com/free-icons/dice>`_.

.. raw:: html

    <embed>
        <p align="center">
            <a href="https://rsc4earth.de/authors/esds/"><img src="https://github.com/davemlz/cubo/raw/main/docs/_static/esds.png" alt="RSC4Earth"></a>
        </p>
    </embed>

