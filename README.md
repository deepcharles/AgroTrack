# AgroTrack

<img align="right" width="200" src="https://github.com/ejalilva/AgroTrack/blob/master/static/AgroTrack.png">

Tracing farmers irrigation decision using satellite observations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ejalilva/AgroTrack
cd AgroTrack
```

2. Install the package and all its dependencies:
```bash
pip install -e . # note the "." at the end
```

AgroTrack is a Python package designed to track farmers' irrigation decisions using thermal remote sensing. It employs hydrological similarity to identify nearby natural pixels and construct a delta LST (Land Surface Temperature) tensor. The package consists of multiple modules, with two key components for tracking farmers' decisions in time and space:

1. **Modules for mapping irrigated areas:**
   - **extract_natural_land_cover_lst:** This function identifies nearby hydrologically similar natural pixels and creates a baseline temperature by averaging their temperatures.
   - **irrigation_mapping_with_deltaLST:** Utilizes the generated Delta LST data cube and a minimum temperature reduction threshold to calculate the annual frequency of irrigation days, focusing exclusively on areas classified as cropland in the MODIS land cover dataset.

2. **Modules for estimating irrigation timing attributes using change point detection:**
   - **irrigation_season_timing:** Creates an array of start, end, and duration of irrigation seasons for each pixel by analyzing the delta LST signal with a binary change point detection algorithm.
   - **irrigation_event_timing:** Applies change point detection to the delta LST time series to identify potential break points caused by irrigation application, cessation, or rate changes. It then uses a segmentation algorithm to identify irrigation episodes.

## Additional features:

- **Data acquisition:** Includes tools to read raw satellite data and create Xarray datasets from NASA servers (for NASA users), including MODIS LST and LAI, SMAP enhanced soil moisture, and MERIT DEM.
- **Integration with Earth Engine:** Utilizes the Xee package, a specialized Xarray extension for Google Earth Engine (GEE), enabling users to work with Earth Engine ImageCollections as Xarray Datasets. This allows streaming of input satellite data without downloading, facilitating the creation of Delta LST Xarray datasets.
- **Change Point Detection**: Apply advanced algorithms from [Ruptures](https://centre-borelli.github.io/ruptures-docs/) package to determine the start and end of irrigation seasons and each individual irrigation events.
- **Parallel processing and data export:** Integrates Dask and Xarray-Beam for parallel processing and exporting datasets to cloud-optimized Zarr format, providing offline accessibility to input data.
- **Result presentation:** Presents irrigation event timing output in an easily interpretable format using SciPy's sparse matrix, ensuring efficient and accessible results.
- **Accuracy evaluation:** Utilizes scikit-learn to build confusion matrices and corresponding evaluation metrics for assessing timing estimation accuracy.

AgroTrack is a satellite data-driven toolbox to track farmers' irrigation decisions over time and across landscapes, helping to better understand the human role in the water cycle.

![AgroTrack Workflow](static/agrotrack_workflow.svg)

*Figure: AgroTrack workflow for tracing farmers' irrigation decisions from thermal satellite data. With just a bounding box and choice of thermal satellite mission (e.g., MODIS, VIIRS, or future thermal constellations), AgroTrack provides a comprehensive range of farmers' irrigation decisions, from strategic planning to operational details.*
