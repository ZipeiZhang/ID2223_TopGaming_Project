### San Francisco Traffic Accident Severity Prediction Project Report

#### Overview
This project aims to predict the severity of traffic accidents in San Francisco using machine learning techniques. Leveraging a large dataset from the San Francisco government’s Open Data Platform, we developed a model to classify traffic incidents into severity categories. This tool is intended to assist in emergency response planning and urban development strategies.

#### Dataset
The primary dataset includes over 800,000 entries from the "Traffic Accidents Reports" section of San Francisco's Open Data Platform. This dataset provides comprehensive details on each traffic incident, such as time, location, and severity of the accidents.

#### Methodology
1. **Data Preprocessing (Notebook 1 - Feature Backfill)**: 
   - The project involved extensive data preprocessing, including filtering relevant columns, mapping severity levels, grouping weather and road conditions, and converting categorical data into numerical formats.
   - Advanced data visualization techniques using libraries like seaborn and matplotlib were applied for in-depth data analysis.

2. **Model Training and Evaluation (Notebook 2 - Training Pipeline)**:
   - We experimented with various machine learning models, including Decision Trees, MLPClassifier, and RandomForestRegressor, focusing primarily on MLPClassifier due to its high performance and stability.
   - Feature engineering was emphasized, creating a robust feature set from the dataset for model training.

3. **Inference Pipeline (Python Script - Inference Pipeline)**:
   - A script was developed to deploy the model into a production environment, facilitating real-time predictions.
   - The script includes functionalities for data ingestion, model loading, and generating predictions.

#### Results
The developed ML model demonstrated promising results in accurately predicting the severity of traffic incidents. By categorizing incidents into severity classes like Minor, Moderate, and Severe, the model provides valuable insights for emergency response units and urban planners.

#### How to Run the Code
1. **Data Preprocessing**:
   - Run the first Jupyter notebook (`project_feature_backfill.ipynb`) to preprocess the dataset. This step involves data cleaning, feature engineering, and data visualization.

2. **Model Training**:
   - Execute the second notebook (`sf_traffic-training-pipeline.ipynb`) to train and evaluate the machine learning models. This notebook includes code for model selection, training, and performance evaluation.

3. **Inference Pipeline**:
   - The Python script (`project_inference_pipeline.py`) can be executed in a production environment. Ensure all dependencies are installed and the script is configured with appropriate access to the model and data sources.

#### Conclusion
The San Francisco Traffic Accident Severity Prediction project successfully demonstrates the application of machine learning in public safety and urban planning. With continuous updates and enhancements, this system can serve as a reliable tool for decision-making in emergency response and city development initiatives.
