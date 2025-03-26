# The EY Global AI and Data Challenge 2025

This repository contains the code and resources for predicting the Urban Heat Island (UHI) index, developed as part of the EY Global AI and Data Challenge 2025. The challenge focuses on addressing the Urban Heat Island (UHI) effect by leveraging AI and data science. I developed a machine learning model that predicts UHI intensity using Sentinel-2 multispectral imagery and building footprint data. It also includes certification documentation validating the project's adherence to challenge guidelines.



## Challenge Overview

The EY Open Science AI & Data Challenge is designed for early-career professionals and university students to combat the Urban Heat Island (UHI) effect—where urban areas can be over 10°C warmer than rural areas due to heavy infrastructure, minimal vegetation, and added anthropogenic heat. The challenge's core objectives are:

- Create a predictive model to detect UHI hotspots using diverse datasets, including ground temperature readings, building footprints, weather data, and satellite imagery (with an emphasis on Sentinel-2).

- Ensure the model is usable by urban planners, with a focus on scalability and socioeconomic impact.

My project uses advanced machine learning methods to predict UHI hotspots in New York City’s Bronx and Manhattan, investigating the key drivers behind these elevated temperatures. This approach aligns with the UN Sustainable Development Goals by promoting sustainable, resilient, and equitable urban development.

## Datasets

In this project, I leveraged the following key datasets:

- **Training Data:**  
  This dataset consists of traverse points across the Bronx and Manhattan. Each record includes longitude, latitude, a timestamp, and the UHI index, serving as the ground truth for model training.

- **Building Footprint Data:**  
  This spatial dataset contains polygon geometries outlining building boundaries in NYC. I used it to extract features such as building counts and total building area within various buffer zones (e.g., 100m, 500m, and 1000m), providing critical context on urban density and structure.

- **Sentinel-2 Satellite Data:**  
  This high-resolution multispectral imagery provides spectral data that I used to derive predictors, including spectral bands (B01–B12, B8A) and indices like NDVI, NDBI, NDWI, and NDMI, which capture details on vegetation, built-up areas, water bodies, and moisture content.

- **Validation Data:**  
  This dataset contains 1040 geographic points (longitude and latitude) without the UHI index, which I used to evaluate the model’s predictive performance on unseen data.


### Urban Heat Island (UHI) Model Training Notebook

In this notebook, I present a baseline workflow that integrates the training data, building footprints and Sentinel-2 imagery to build a model for predicting the UHI index.

1. **Data Preparation and Preprocessing:**  
   - **Training Data:** I load training data that include UHI index measurements.
   - **Building Footprint Data:** I incorporate detailed building geometries and extract features such as building counts and total building area within multiple buffer zones (e.g., 100m, 500m, and 1000m).
   - **Sentinel-2 Data:** I process the spectral bands and compute indices.

2. **Feature Engineering:**  
   - I combine building-related features with spectral predictors derived from Sentinel-2 data.
   - I then prepare a unified feature set for model training.  
   - The features used in my model include:  
     `['building_count_100m', 'total_building_area_100m', 'building_count_500m', 'total_building_area_500m', 'building_count_1000m', 'total_building_area_1000m', 'area_rural', 'area_urban', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B12', 'B8A', 'NDVI', 'NDBI', 'NDWI', 'NDMI']`

### Model Training

I developed the UHI predictive models using several machine learning algorithms to identify the most effective approach for predicting the UHI index. The training process involved:

- **Data Integration:**  
  Merging the training data with engineered features derived from building footprints and Sentinel-2 imagery to create a unified dataset.

- **Preprocessing:**  
  Normalizing the data and ensuring high-quality input for the models.

- **Model Selection and Hyperparameter Tuning:**  
  Experimenting with different algorithms—including Random Forest, XGBoost, and ExtraTrees—and tuning parameters such as the number of trees and maximum tree depth to optimize performance.

- **Training and Evaluation:**  
  Training each model and evaluating their performance based on various metrics to determine the best predictor for the UHI index.

### Model Evaluation

The table below summarizes the test results for the UHI predictive models. The metrics include the R² Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

| Model         | R² Score | MSE (×10⁻⁵) | RMSE   | MAE   |
|---------------|---------------|------------------|-------------|------------|
| Random Forest | 0.9255        | 1.96             | 0.00443     | 0.00305    |
| XGBoost       | 0.9263        | 1.94             | 0.00440     | 0.00319    |
| ExtraTrees    | 0.9425        | 1.51             | 0.00389     | 0.00261    |


These test results illustrate that all models are capable of capturing the key factors influencing the UHI effect, with the ExtraTrees model showing the best performance overall.

### Conclusion

This project represents my approach to tackling the Urban Heat Island (UHI) effect using AI, open-source data, and machine learning. By integrating Sentinel-2 imagery with building footprint data, I developed predictive models that effectively identify UHI hotspots and provide actionable insights for urban planners, demonstrating how advanced AI & Data techniques can contribute to sustainable urban development.

For more details, please refer to the code and documentation in this repository.

### Repository Contents

- `data/`: Contains the datasets used for analysis.
- `notebooks/`: Includes Jupyter notebooks for exploratory data analysis and model development.
- `submission/`: Holds the predicted UHI index for the validation data.
- `requirements.txt`: Python dependencies and packages required to run the project.
- `README.md`: Provides an overview of the project and documentation.
