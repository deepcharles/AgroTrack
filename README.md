# AgroTrack

<img align="right" width="200" src="https://github.com/ejalilva/AgroTrack/blob/master/AGRotrack.png">

tracing farmers irrigation decision using satellite observations

AgroTrack is a Python package designed to trace farmers' irrigation decisions using satellite-observed thermal signatures. By leveraging advanced satellite data processing techniques, AgroTrack offers precise and scalable tools for mapping irrigated areas, detecting the start and end of the irrigation season and individual irrigation events.


## Features

- **Delta LST Calculation**: Utilize MODIS land surface temperature data to identify irrigation events based on the temperature difference between irrigated and natural pixels.
- **Change Point Detection**: Apply advanced algorithms to determine the start and end of irrigation seasons and individual irrigation events.
- **Integration with Google Earth Engine**: Use the XEE (Xarray extension for GEE) to manage and process large satellite datasets.
- **Scalability**: Leverage Dask and Xarray-Beam for efficient parallel processing.
- **Evaluation Metrics**: Implement confusion matrix and evaluation metrics using sci-kit to assess timing estimation accuracy.

## Installation

To install AgroTrack, use the following command:

```bash
pip install agrotrack
