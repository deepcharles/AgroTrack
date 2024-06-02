# AgroTrack

<img align="right" width="200" src="https://github.com/ejalilva/AgroTrack/blob/master/static/AgroTrack.png">

tracing farmers irrigation decision using satellite observations

AgroTrack is a Python package designed to trace farmers' irrigation decisions using satellite-observed thermal signatures. By leveraging thermal satellite data, AgroTrack offers precise and scalable tools for mapping irrigated areas, detecting the start and end of the irrigation season and individual irrigation events.


## Features

- **Delta LST Calculation**: Utilize MODIS land surface temperature data to identify irrigation events based on the temperature difference between irrigated and natural land covers.
- **Change Point Detection**: Apply advanced algorithms from Ruptures package to determine the start and end of irrigation seasons and each individual irrigation events.
- **Integration with Google Earth Engine**: Use the XEE (Xarray extension for GEE) to manage read the data from GEE catalog and create a Xarray datacube.
- **Scalability**: Leverage Dask and Xarray-Beam for efficient parallel processing.
- **Evaluation Metrics**: Implement confusion matrix and evaluation metrics using sci-kit to assess timing estimation accuracy.

## Installation

To install AgroTrack, use the following command:

```bash
pip install agrotrackThere a
