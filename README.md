# Shell AI Hackathon 2023

**Biomass Forecasting and Supply Chain Optimization for the Shell.ai Hackathon 2023**

## Getting Started

To get started, make sure you have Python 3.10 installed and follow these steps:

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Set the notebook kernel to the right environment.

## Forecasting Methodology

### Generate Forecast

You can find the code for generating the biomass forecast in the [**generate_forecast.ipynb**](notebooks/generate_forecast.ipynb) notebook.

#### Biomass Dataset Cleanup

To ensure data accuracy, the notebook addresses and fills in duplicated values that occurred before the 2014 census.

#### Cluster Indexes

The clustering process is primarily based on district names, followed by checking correlations for each index within each district. Each index is assigned to the district with the highest Pearson correlation.

#### Create Table for Crop Production

A table containing crop production data for each district is created based on [Desagri data](https://data.desagri.gov.in/website/crops-report-major-contributing-district-web). Missing values before 2014 are filled in using production conservation ratios.

#### Add Elevation Map and Crop Land Map

Crop land data from [EarthStat](http://www.earthstat.org/) and elevation data from NASA Earth Observation [NEO](https://neo.gsfc.nasa.gov/dataset_index.php#energy) are integrated into the analysis.

#### Train Model

The model pipeline consists of a MaxAbsScaler and an ExtraTreeRegressor. Cross-validation is performed for each year based on all other years, with the following results:

| Year | Test MAE |
| ---- | -------- |
| 2010 | 22.6     |
| 2011 | 19.4     |
| 2012 | 27.7     |
| 2013 | 32.9     |
| 2014 | 24.9     |
| 2015 | 20.8     |
| 2016 | 29.1     |
| 2017 | 29.6     |
| Avg  | 25.9     |

#### Inference on 2018 and 2019

The model, trained on historical data, is used for inference on 2018 and 2019, and the forecast is stored for further use in the optimization step.

![Biomass Forecast](./docs/forecast_img.PNG)

## Optimization Methodology

### Generate Optimized Locations

The code for generating optimized locations can be found in the [**generate_optimized_locations.ipynb**](notebooks/generate_optimized_locations.ipynb) notebook.

#### Refineries Location

The number of refineries is defined to collect 80% of biomass production (a problem constraint). Initial refinery positions are set at the center of the main biomass clusters.

#### Initial Depot Location by Subtraction

The process includes the following steps, repeated until a maximum iteration is reached:

1. Start with around 60 depots spread in regions with high biomass (>200) in a random manner.

2. Calculate the flux to refineries using linear optimization.

3. Remove the depot that is the most underutilized.

4. Stop if constraints cannot be satisfied when calculating flux from depots to refineries.

The final depot positions giving the best cost over all runs are extracted.

#### Fine-Tune Depot/Refineries Positions Using a Greedy Algorithm

For each depot + refinery index:

1. Calculate costs in multiple directions for an initial distance.

2. Keep the new position if the cost is lower.

If the cost is not improved after a full loop, the distance is increased.

#### Final Depot and Refineries Locations (Optimized on 2019)

![Optimized Locations](./docs/optimized_locations_img.PNG)

## Scoring on 2018 and 2019 for the Final Submission

| Year | Forecast MAE | Optimization Cost | Score  |
| ---- | ------------ | ------------------ | ------ |
| 2018 | 24.39        | 44,150             | 83.49  |
| 2019 | 30.69        | 26,786             | 83.84   |

Note that the optimization cost on 2018 is high as the same infrastructure had to be used for both years, the final submission is optimized on 2019.
