# San Francisco Traffic Accident Severity Prediction

## Overview
This project is focused on the classification of traffic accident severity in San Francisco using a machine learning model. By leveraging detailed historical data, the goal is to provide a predictive insight into the severity of traffic incidents to assist in emergency response and urban planning.

## Dataset
The dataset utilized for this project is derived from the "Traffic Accidents Reports" section of the San Francisco government's Open Data Platform, including a wide range of traffic incident details such as time, severity, location, and more. The dataset is regularly updated and at the time of the model's last training, contained over 800,000 entries.

To access the dataset, an App Token is required along with the installation of the `sodapy` package. The full historical data can be retrieved by setting `limit=800000` in the API call.

## Methodology

### Data Preprocessing
Data preprocessing is a critical step to ensure the dataset is fit for model training. The following steps were conducted:

1. **Subset Selection**: We filtered the dataset to include only the columns pertinent to our predictive analysis.

2. **Severity Mapping**: Incident severities were categorized into classes such as 'Minor', 'Moderate', and 'Severe', including fatalities.

3. **Weather Grouping**: The diverse weather conditions were consolidated into more general categories.

4. **Categorical Mapping**: Numerical mappings were assigned to categorical variables, transforming them into a machine-readable format.

### Feature Engineering
The dataset was transformed to create a feature set which included:

- Temporal aspects of the incident (year, month, hour)
- Weather conditions
- Road and lighting conditions
- Collision types
- Movements and types of involved parties
- The target variable: the incident severity class.

### Model Training
The model was trained using the XGBoost algorithm to classify the severity of traffic incidents based on the features. We used an 80-20 split for training and testing sets and aimed to introduce data balancing to enhance model performance.

## Deliverables

### Interactive Prediction UI
An interactive UI allows users to input incident features and receive a prediction for the incident's severity.

### Dashboard Monitoring UI
A dashboard UI displays the latest predictions on traffic accident severities and their actual values, updating in real-time as new data becomes available.

## Model Monitor UI
Utilizing Hugging Face Spaces, this UI visualizes the latest incidents with both predicted and actual severities. It demonstrates the model's real-world applicability in predicting the severity of traffic accidents.

## Interactive UI
This UI, also powered by Hugging Face Spaces, engages users by providing real-time severity predictions based on the details of the incident they input.

## Conclusion
Our model provides a valuable resource for understanding and predicting traffic accident severities in San Francisco, potentially aiding in better emergency response and city planning strategies. The model's predictive accuracy indicates its utility, with room for future enhancements.
