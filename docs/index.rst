.. toctree::
   :maxdepth: 2
   :hidden:
  
   reference
   tutorials
   changelog

Cubo
====

Overview
--------

SpatioTemporal Asset Catalogs (STAC) provide a standardized format that describes
geospatial information. Multiple platforms are using this standard to provide clients 
several datasets. Nice platforms such as Planetary Computer use this standard.

`cubo` is a Python package that provides users of STAC objects an easy way to create 
Earth Observation (EO) mini cubes. This is perfectly suitable for Machine Learning (ML) / 
Deep Learning (DL) tasks. You can easily create a lot of mini cubes by just knowing a pair 
of coordinates and the edge size of the cube in pixels!

Check the simple usage of `cubo` here:

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
4. Additional information is used to retrieve the data from the STAC catalogue: starts 
and end dates, name of the collection, endpoint of the catalogue, etc.
5. Then, by using `stackstac` and `pystac_client` the mini cube is retrieved as a 
`xr.DataArray`.
6. Success! That's what `cubo` is doing for you, and you just need to provide the 
coordinates, the edge size, and the additional info to get the cube.

Installation
------------

Install the latest version from PyPI:

.. code-block::

   pip install cubo


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
~~~~~~~

The project is licensed under the MIT license.

