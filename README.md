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

<h2>Exploratory Data Analysis</h2>

[EDA Notebook](https://github.com/allankapoor/wildfire_prediction/blob/master/Step2_ExploratoryDataAnalysis.ipynb)

<h2>Preprocessing</h2>

[Preprocessing Notebook](https://github.com/allankapoor/wildfire_prediction/blob/master/Step3_Preprocessing.ipynb)

Prior to modelling, transformations were applied to the continuous explanatory variables in order to reduce skew and bring their distributions as close to normal as possible.</p>

Aspect (i.e. the cardinal direction a slope faces in degrees from north) and discovery day of the year are actually cyclical features in that their values “wrap around” - the highest values are close to the lowest values. In order for this nuance to be apparent to the models, these features were both transformed into dual harmonic variables that swing back and forth out of sync.</p>

<h2>Modelling</h2>

[Modeling Notebook](https://github.com/allankapoor/wildfire_prediction/blob/master/Step4_Modeling.ipynb)

I tested several different models, each with and without oversampling. The primary evaluation metrics was F2 score. While the F1 score is the harmonic mean of precision and recall, the F2 score calculates the harmonic mean with an additional coefficient that essentially weights recall higher than precision.</p>

Models were evaluated using 10-fold cross validation and tuned with RandomSearchCV (100 permutations). The best performing model (LightGBM) was optimized further via Optuna (200 trials). This model achieves an F2 score of 0.297 and an ROC AUC of 0.691 on test data.</p>

The performance of each model is presented in the table below. The columns to the left summarize the mean results of the 10-fold cross validation and the columns to the right display results when the model was trained on the full cross validation set and then tested on a single validation set (not the test set which is only used on the final selected model.</p>

<h1>Conclusions</h1>

While the final model does have substantial predictive power, it could certainly be improved. Some ideas for further refinement:</p>

 * The timeframes for the weather features calculated from Google Earth Engine could be revisited. In particular, the timeframe for precipitation (previous year) could be shortened.
* Additional features that address human activity/influence could be added. For example, distance from paved roads or distance from CALFIRE airports. 
The categorical vegetation type and ecoregion datasets did not end up having as strong of predictive power as anticipated. These could be replaced or supplemented with more granular quantitative datasets such as Normalized Difference Vegetation Index (NDVI) (for the days preceding each wildfire), canopy density, fuel load, etc.
* The model may also be suffering from not having enough examples of the positive minority class to train on. This could be addressed by using updated data that extends to 2018 (rather than 2015). This updated dataset was unfortunately released after the feature extraction phase of this project was complete. Another possibility is to extend the start date back from 2005 to 2000, or as far as 1992.
* The class imbalance could also be addressed by reducing the scope of the model. The model could be limited to months in summer and early fall (when large wildfires actually occur) or to wildfire prone areas, rather than the entire state. This might ensure that the training data is more directly relevant to the desired use case for the model.

While this model was evaluated based on a hold-out test set split from a dataset of historic wildfires, the purpose of this model is to make predictions for future wildfires as they occur. The model could be put into production with a front-end interface where the user could indicate the location of a wildfire on an interactive browser-based map, enter the date, and then receive a prediction. For situations where many wildfires are occurring at once, a spatial file (shapefile, geojson, etc.) or table of wildfire locations could be uploaded in order for the model to make batch predictions.

<h2>Credits</h2>
Thanks to Shmuel Naaman for mentorship and advice on feature engineering/algorithms, and to Diana Edwards for help thinking through relevant explanatory features, appropriate time frames for weather variables, and sources for vegetation data.
