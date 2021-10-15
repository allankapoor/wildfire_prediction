![Wildfires](https://github.com/allankapoor/wildfire_prediction/blob/master/Images/Cover.png)

# Predicting Wildfire Severity in California Leveraging Google Earth Engine and Machine Learning

This project is in effort to predict wildfire severity based on the location, time of the year, and environmental features of historic wildfires leveraging machine learning algorithms. Wildfire records (location, date, and burned area) were sourced from the [United States Forest Service (USFS)](https://doi.org/10.2737/RDS-2013-0009.4). These records were enriched with additional features from several spatiotemporal datasets via Google Earth Engine and then used to train a series of machine learning algorithms to predict, if given a location and date of discovery, a wildfire will burn greater than 300 acres. 

View the [presentation](https://github.com/allankapoor/wildfire_prediction/blob/master/PRESENTATION.pdf), full [report](https://github.com/allankapoor/wildfire_prediction/blob/master/REPORT.pdf), or check out the summary below.

<h2>Why wildfire prediction?</h2>

Wildfires are a major natural hazard in California and the severity of wildfires has increased substantially in recent years. A model that predicts which fires have the potential to become most severe would enable responders to make more informed decisions about how to allocate limited resources, protecting lives, property, and the environment.</p>

<h2>Challenges</h2>

1. __No explanatory features:__ the primary dataset of wildfire records only includes the location, date, and final burned area of each wildfire. All explanatory features had to be extracted from various spatiotemporal weather datasets or spatial data on topography, vegetation type, and habitat.
2. __Localized factors__: Wildfire hazard is highly localized and time specific. Weather and environmental data needed to have both high spatial granularity and high temporal granularity. This required leveraging Google Earth Engine’s cloud computing resources to handle large amounts of daily gridded time series.
3. __Imbalanced data__: large wildfires (the positive class) make up a very small portion of the dataset. I attempted to address this through class weighting and SMOTE oversampling. 

<h1> Project Summary </h1>
<h2>Data Wrangling + Feature Extraction</h2>

[Data Wrangling + Feature Extraction Notebook](https://github.com/allankapoor/wildfire_prediction/blob/master/Step1_DataWrangling-FeatureGeneration.ipynb)


| Variable                       | Time frame     | Description                                                                                                                                                                                                                        | Source                                                                                                                                                                                                            | Format                               |
| ------------------------------ | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| Elevation                      | n/a              | Vertical elevation above sea level (NAVD 88), meters                                                                                                                                                                               | [USGS National Elevation Dataset](https://developers.google.com/earth-engine/datasets/catalog/USGS_NED) (via Google Earth Engine)                                                                                 | 2D grid, 10.2 m resolution           |
| Slope                          | n/a              | Vertical slope, degrees                                                                                                                                                                                                            | Calculated from elevation dataset                                                                                                                                                                                 | 2D grid, 10.2 m resolution           |
| Aspect                         | n/a              | Direction of slope face, degrees from North (clockwise)                                                                                                                                                                            | Calculated from elevation dataset                                                                                                                                                                                 | 2D grid, 10.2 m resolution           |
| Temperature                    | Preceding 7 days | Maximum daily temperature, °C                                                                                                                                                                                                      | [PRISM](https://developers.google.com/earth-engine/datasets/catalog/OREGONSTATE_PRISM_AN81d?hl=en#bands) Daily Spatial Climate Dataset (“tmax” band)                                                              | Gridded time series, 4 km resolution |
| Dew point                      | Preceding 7 days | Daily mean dew point temperature - a measure of air moisture, °C                                                                                                                                                                   | [PRISM](https://developers.google.com/earth-engine/datasets/catalog/OREGONSTATE_PRISM_AN81d?hl=en#bands) Daily Spatial Climate Dataset (“dtmean” band)                                                            | Gridded time series, 4 km resolution |
| Precipitation                  | Preceding year   | Monthly precipitation, millimeters                                                                                                                                                                                                 | [PRISM](https://developers.google.com/earth-engine/datasets/catalog/OREGONSTATE_PRISM_AN81m) Monthly Spatial Climate Dataset (“ppt” band)                                                                         | Gridded time series, 4 km resolution |
| Wind speed                     | Day of discovery | Wind speed,  meters per second                                                                                                                                                                                                     | [GRIDMET](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET?hl=en#description): University of Idaho Gridded Surface Meteorological Dataset (“vs” band) via Google Earth Engine     | Gridded time series, 4 km resolution |
| Energy Release Component (ERC) | Day of discovery | The ERC is an index related to the available energy (BTU) per unit area (square foot) within the flaming front at the head of a fire. Each daily calculation considers the past 7 days.                                           | [GRIDMET](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET?hl=en#description): University of Idaho Gridded Surface Meteorological Dataset (“erc” band) via Google Earth Engine    | Gridded time series, 4 km resolution |
| Burning index (BI)             | Day of discovery | A measure of fire intensity. BI has no units, but in general it is 10 times the flame length of a fire.                                                                                                                            | [GRIDMET](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET?hl=en#description): University of Idaho Gridded Surface Meteorological Dataset (“bi” band) via Google Earth Engine     | Gridded time series, 4 km resolution |
| 100-hour dead fuel moisture    | Day of discovery | Represents the modeled moisture content of dead fuels in the 1 to 3 inch diameter class. Values can range from 1 to 50 percent.                                                                                                    | [GRIDMET](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET?hl=en#description): University of Idaho Gridded Surface Meteorological Dataset (“fm100” band) via Google Earth Engine  | Gridded time series, 4 km resolution |
| 1000-hour dead fuel moisture   | Day of discovery | Represents the modeled moisture content in dead fuels in the 3 to 8 inch diameter class. Values can range from 1 to 40 percent.                                                                                                    | [GRIDMET](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET?hl=en#description): University of Idaho Gridded Surface Meteorological Dataset (“fm1000” band) via Google Earth Engine | Gridded time series, 4 km resolution |
| Vegetation Type                | n/a              | Vegetation types, compiled from a variety of state/federal sources into a single comprehensive data set                                                                                                                            | [CALFIRE](https://map.dfg.ca.gov/metadata/ds1327.html) Forest and Rangeland Assessment                                                                                                                            | 2D grid, 30 m resolution             |
| Level III Ecoregions           | n/a              | Ecoregions are areas where ecosystems (and the type, quality, and quantity of environmental resources) are generally similar.                                                                                                      | [Ecoregions of the Continental United States](https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states), US EPA                                                                     | Shapefile (polygon vector)           |
| Burn probability               | n/a              | Output from the FSim probabilistic wildfire model. The burn probability dataset is the simulated mean annual burn probability.                                                                                                     | [Wildfire Hazard Potential for the United States](https://www.fs.usda.gov/rds/archive/Catalog/RDS-2015-0047-3), US Forest Service                                                                                 | 2D grid, 270 m resolution            |
| Fire intensity level (1-6)     | n/a              | Output from the FSim probabilistic wildfire model. The fire intensity level dataset consists of six raster files, each representing the portion of all simulated fires that burned in the cell area at the specified flame length. | [Wildfire Hazard Potential for the United States](https://www.fs.usda.gov/rds/archive/Catalog/RDS-2015-0047-3), US Forest Service                                                                                 | 2D grid, 270 m resolution            |


<h2>Exploratory Data Analysis</h2>

[EDA Notebook](https://github.com/allankapoor/wildfire_prediction/blob/master/Step2_ExploratoryDataAnalysis.ipynb)

_The majority of wildfires are small, but large wildfires cause most of the damage._ Out of the 83,606 wildfires recorded in California 2005-2015, 51% each burned 0.25 acres or less (about a third the size of an American football field). Only 1.2% of the wildfires burned 300 acres or more. However, that 1.2% of wildfires contributed to 96% of the total area burned.</p>
![Wildfire size](https://github.com/allankapoor/wildfire_prediction/blob/master/Images/WildfireSizeBar.png)

_During peak wildfire season, hundreds of wildfires can start on the same day._ The figure below displays the number of wildfires discovered in California each day from 2005 to 2015. Seasonal oscillations are apparent, with new wildfires per day peaking in the summer. There are 33 different days when 100 or more new fires were discovered in a single day.</p>
![Wildfire Frequency](https://github.com/allankapoor/wildfire_prediction/blob/master/Images/WildfireFrequency.png)

_There is a severe class imbalance between small and large wildfires._ While an ideal model would be able to predict the size class of a given wildfire, due to the low number of records of the larger size classes, a binary classification model will likely achieve better results and still be useful for wildfire prioritization. A wildfire that could burn more than 300 acres would definitely be of concern. </p>
![Wildfire Size Class](https://github.com/allankapoor/wildfire_prediction/blob/master/Images/WildfireSizeClass.png)

For all continuous variables, I did visual EDA and two-tailed t-tests vs. wildfire size. In general, the means for most of the variables have a significant difference (with low p values) between for small and large wildfires, suggesting that these variables will have predictive power during modelling. Here is one example:</p>
![Dew Point](https://github.com/allankapoor/wildfire_prediction/blob/master/Images/DewPointEDA.png)

<h2>Preprocessing</h2>

[Preprocessing Notebook](https://github.com/allankapoor/wildfire_prediction/blob/master/Step3_Preprocessing.ipynb)

Prior to modelling, transformations were applied to the continuous explanatory variables in order to reduce skew and bring their distributions as close to normal as possible.</p>

Aspect (i.e. the cardinal direction a slope faces in degrees from north) and discovery day of the year are actually cyclical features in that their values “wrap around” - the highest values are close to the lowest values. In order for this nuance to be apparent to the models, these features were both transformed into dual harmonic variables that swing back and forth out of sync.</p>

<h2>Modelling</h2>

[Modeling Notebook](https://github.com/allankapoor/wildfire_prediction/blob/master/Step4_Modeling.ipynb)

I tested several different models, each with and without oversampling. The primary evaluation metrics was F2 score. While the F1 score is the harmonic mean of precision and recall, the F2 score calculates the harmonic mean with an additional coefficient that essentially weights recall higher than precision.</p>

Models were evaluated using 10-fold cross validation and tuned with RandomSearchCV (100 permutations). The best performing model (LightGBM) was optimized further via Optuna (200 trials). This model achieves an F2 score of 0.297 and an ROC AUC of 0.691 on test data.</p>

The performance of each model is presented in the table below. The columns to the left summarize the mean results of the 10-fold cross validation and the columns to the right display results when the model was trained on the full cross validation set and then tested on a single validation set (not the test set which is only used on the final selected model.</p>

| Model Performance      | f2    | recall | roc auc |
| ---------------------- | ----- | ------ | ------- |
| LGBM (Optuna)          | 0.296 | 0.457  | 0.688   |
| LightGBM               | 0.280 | 0.447  | 0.680   |
| XGBoost                | 0.260 | 0.550  | 0.699   |
| Random Forest          | 0.255 | 0.413  | 0.661   |
| LightGBM w/ SMOTE      | 0.248 | 0.476  | 0.674   |
| Random Forest w/ SMOTE | 0.240 | 0.507  | 0.678   |
| XGB (SMOTE)            | 0.239 | 0.493  | 0.674   |
| Logistic Regression    | 0.232 | 0.748  | 0.731   |
| Dummy Model            | 0.025 | 0.025  | 0.500   |

![ROC AUC chart](https://github.com/allankapoor/wildfire_prediction/blob/master/Images/ROC_AUC.png)

<h1>Conclusions</h1>

Some ideas for further refinement:</p>

 * The timeframes for the weather features calculated from Google Earth Engine could be revisited. In particular, the timeframe for precipitation (previous year) could be shortened.
* Additional features that address human activity/influence could be added. For example, distance from paved roads or distance from CALFIRE airports. 
The categorical vegetation type and ecoregion datasets did not end up having as strong of predictive power as anticipated. These could be replaced or supplemented with more granular quantitative datasets such as Normalized Difference Vegetation Index (NDVI) (for the days preceding each wildfire), canopy density, fuel load, etc.
* The model may also be suffering from not having enough examples of the positive minority class to train on. This could be addressed by using updated data that extends to 2018 (rather than 2015). This updated dataset was unfortunately released after the feature extraction phase of this project was complete. Another possibility is to extend the start date back from 2005 to 2000, or as far as 1992.
* The class imbalance could also be addressed by reducing the scope of the model. The model could be limited to months in summer and early fall (when large wildfires actually occur) or to wildfire prone areas, rather than the entire state. This might ensure that the training data is more directly relevant to the desired use case for the model.

Using the model:</p>
While this model was evaluated based on a hold-out test set split from a dataset of historic wildfires, the purpose of this model is to make predictions for future wildfires as they occur. The model could be put into production with a front-end interface where the user could indicate the location of a wildfire on an interactive browser-based map, enter the date, and then receive a prediction. For situations where many wildfires are occurring at once, a spatial file (shapefile, geojson, etc.) or table of wildfire locations could be uploaded in order for the model to make batch predictions.

<h2>Credits</h2>
Thanks to Shmuel Naaman for mentorship and advice on feature engineering/algorithms, and to Diana Edwards for help thinking through relevant explanatory features, appropriate time frames for weather variables, and sources for vegetation data.
