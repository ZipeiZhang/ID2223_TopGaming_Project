#San Francisco Traffic Accident Severity Prediction
Overview
This project focuses on predicting the severity of traffic accidents in San Francisco using a machine learning approach. By analyzing historical traffic accident data from the city's open data platform, we aim to classify incidents into severity classes to aid in emergency services response and city planning.

Dataset
The dataset is sourced from the "Traffic Accidents Reports" section of the San Francisco government's Open Data Platform. It includes detailed traffic incident records, encompassing attributes like time, severity, and location. The dataset is continually updated and contains over 800,000 entries to date.

Accessing the dataset requires an App Token registration and the installation of the sodapy package. We set limit=800000 to ensure we capture all historical data available at the time of our data fetch.

Methodology
Data Preprocessing
The preprocessing stage is crucial to streamline the dataset for model consumption. We initiated this by selecting only relevant columns and discarding entries with missing values to maintain data integrity. The code snippet provided outlines the preprocessing steps that were taken:

We first created a subset of the dataframe containing only the columns relevant to our analysis.
Severity mapping was applied to categorize the severity levels into 'Minor', 'Moderate', and 'Severe' injuries, including fatal incidents.
Weather conditions were grouped into broader categories to reduce fragmentation and potential overfitting due to rare occurrences.
Numerical mappings were assigned to categorical variables such as weather conditions, intersection types, road conditions, and other features, enabling the model to process them effectively.
This process ensured that the dataset was clean, manageable, and ready for the feature engineering phase.

Feature Engineering
We transformed the dataset into a feature set comprising:

Time of incident (year, month, hour)
Weather conditions
Road conditions
Lighting conditions
Type of collision
Movements and types of parties involved in the accident
The target variable for prediction was the severity class of the incident.
Model Training
A predictive model was trained using the XGBoost algorithm to classify the severity of traffic incidents. The dataset was split into training and testing sets with an 80-20 ratio. The trained model achieved a certain level of accuracy, indicating its ability to generalize from the provided features.

Deliverables
We provided two key deliverables:

Interactive Prediction UI: A user-friendly interface that allows users to input incident details and obtain a severity prediction.

Dashboard Monitoring UI: A real-time monitoring dashboard displaying recent traffic incidents and their predicted severities, alongside actual severities.

Predictive Model Monitor UI
The monitoring UI, powered by Hugging Face Spaces, visualizes the latest traffic incidents with predicted and actual severities. It fetches and processes data regularly, displaying predictions from the trained model in an accessible manner.

Interactive Prediction UI
This interface leverages Hugging Face Spaces to offer real-time severity predictions based on user-provided incident details. The model, hosted on Hopsworks, visualizes the severity to enhance user understanding of potential traffic risks.

Conclusion
Our predictive system offers a valuable tool for assessing traffic accident severity in San Francisco. By providing insights into incident severities, the model can facilitate more effective emergency responses and inform traffic management decisions. Future improvements may include incorporating additional data sources or adopting advanced modeling techniques to refine prediction accuracy.
