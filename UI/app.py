import gradio as gr
import hopsworks
import joblib
import pandas as pd

api = 'UtYWT9JBE4jbsOVW.dzfTExU7QMCzzR51EADTOZCXBzl0VmgB2y012yd8nFTG6v1VHgWazdx2a2SuJAY1'
project = hopsworks.login(api_key_value = api)


mr = project.get_model_registry()
model = mr.get_model("sf_traffic_model_1", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/sf_traffic_model_1.pkl")
print("Model downloaded")
    
# Mapping functions for each categorical feature
def map_time_cat(time_cat):
    time_cat_mapping = {
        '2:01 pm to 6:00 pm': 0, '10:01 am to 2:00 pm': 1, '6:01 pm to 10:00 pm': 2, 
        '6:01 am to 10:00 am': 3, '10:01 pm to 2:00 am': 4, '2:01 am to 6:00 am': 5
    }
    return time_cat_mapping.get(time_cat, -1)  # Default to -1 for unknown categories

def map_party2_move_pre_acc(move):
    party2_move_mapping = {
        'Proceeding Straight': 0, 'Stopped In Road': 1, 'Not Stated': 2, 'Making Left Turn': 3,
        'Parked': 4, 'Other': 5, 'Stopped': 6, 'Making Right Turn': 7, 'Slowing/Stopping': 8,
        'Entering Traffic': 9, 'Backing': 10, 'Changing Lanes': 11, 'Parking Maneuver': 12,
        'Passing Other Vehicle': 13, 'Making U Turn': 14, 'Merging': 15, 'Ran Off Road': 16,
        'Traveling Wrong Way': 17, 'Crossed Into Opposing Lane': 18,
        'Other Unsafe Turning': 19, 'Crossed Into Opposing Lane - Unplanned': 20
    }
    return party2_move_mapping.get(move, -1)

def map_party1_move_pre_acc(move):
    party1_move_mapping = {
        'Proceeding Straight': 0, 'Making Left Turn': 1, 'Making Right Turn': 2, 
        'Changing Lanes': 3, 'Other': 4, 'Entering Traffic': 5, 'Not Stated': 6, 'Backing': 7, 
        'Stopped In Road': 8, 'Making U Turn': 9, 'Parked': 10, 'Slowing/Stopping': 11, 
        'Passing Other Vehicle': 12, 'Traveling Wrong Way': 13, 'Stopped': 14, 'Ran Off Road': 15, 
        'Other Unsafe Turning': 16, 'Parking Maneuver': 17, 'Crossed Into Opposing Lane': 18, 
        'Merging': 19, 'Crossed Into Opposing Lane - Unplanned': 20
    }
    return party1_move_mapping.get(move, -1)

def map_ped_action(action):
    ped_action_mapping = {
        'No Pedestrian Involved': 0, 'Crossing in Crosswalk at Intersection': 1,
        'Crossing Not in Crosswalk': 2, 'In Road, Including Shoulder': 3, 'Not in Road': 4,
        'Not Stated': 5, 'Crossing in Crosswalk Not at Intersection': 6,
        'Approaching/Leaving School Bus': 7
    }
    return ped_action_mapping.get(action, -1)

def map_party1_type(party_type):
    party1_type_mapping = {
        'Driver': 0, 'Bicyclist': 1, 'Pedestrian': 2, 'Other': 3, 'Parked Vehicle': 4, 
        'Not Stated': 5
    }
    return party1_type_mapping.get(party_type, -1)

def map_party2_type(party_type):
    party2_type_mapping = {
        'Driver': 0, 'Pedestrian': 1, 'Bicyclist': 2, 'Parked Vehicle': 3, 'Other': 4, 
        'Not Stated': 5
    }
    return party2_type_mapping.get(party_type, -1)

def map_mviw(mviw):
    mviw_mapping = {
        'Other Motor Vehicle': 0, 'Pedestrian': 1, 'Bicycle': 2, 'Fixed Object': 3, 
        'Parked Motor Vehicle': 4, 'Non-Collision': 5, 'Not Stated': 6, 'Other Object': 7, 
        'Motor Vehicle on Other Roadway': 8, 'Train': 9, 'Animal': 10
    }
    return mviw_mapping.get(mviw, -1)


def map_intersection(intersection):
    mapping = {
        'Intersection <= 20ft': 0,
        'Midblock > 20ft': 1,
        'Intersection Rear End <= 150ft': 2
    }
    return mapping.get(intersection, -1)

def map_road_cond_1(road_cond):
    mapping = {
        'No Unusual Condition': 0,
        'Not Stated': 1,
        'Other': 2,
        'Construction or Repair Zone': 3,
        'Holes, Deep Ruts': 4,
        'Obstruction on Roadway': 5,
        'Loose Material on Roadway': 6,
        'Holes, Deep Rut': 7,
        'Reduced Roadway Width': 8,
        'Flooded': 9
    }
    return mapping.get(road_cond, -1)

def map_control_device(device):
    mapping = {
        'Functioning': 0,
        'None': 1,
        'Not Stated': 2,
        'Not Functioning': 3,
        'Obscured': 4
    }
    return mapping.get(device, -1)

def map_lighting(lighting):
    mapping = {
        'Daylight': 0,
        'Dark - Street Lights': 1,
        'Dusk - Dawn': 2,
        'Not Stated': 3,
        'Dark - No Street Lights': 4,
        'Dark - Street Lights Not Functioning': 5
    }
    return mapping.get(lighting, -1)

def map_road_surface(surface):
    mapping = {
        'Dry': 0,
        'Wet': 1,
        'Not Stated': 2,
        'Slippery': 3,
        'Snowy or Icy': 4
    }
    return mapping.get(surface, -1)

def map_type_of_collision(collision):
    mapping = {
        'Broadside': 0,
        'Vehicle/Pedestrian': 1,
        'Rear End': 2,
        'Sideswipe': 3,
        'Head-On': 4,
        'Other': 5,
        'Hit Object': 6,
        'Not Stated': 7,
        'Overturned': 8
    }
    return mapping.get(collision, -1)

def map_weather_condition(conditions):
    # If 'conditions' is a list with one element, take the first element
    if isinstance(conditions, list) and len(conditions) == 1:
        condition = conditions[0]
    else:
        condition = conditions  # Otherwise, assume it's a single value
    
    weather_mapping = {
        'Clear': 0, 
        'Cloudy': 1, 
        'Raining': 2, 
        'Fog': 3, 
        'Other': 4, 
        'Not Stated': 5, 
        'Unknown': 6
    }
    return weather_mapping.get(condition, -1)  # Default to -1 for unknown categories

def update_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sodapy import Socrata
    import hopsworks
    import pandas as pd
    from sodapy import Socrata
    from data_preprocessing import data_preprocess
    # Load the data
    client = Socrata("data.sfgov.org", "7cCZMHDK3CTLNu0K31rZxP1bl", username="zzp9067788@gmail.com", password="Zzp123456.")
    results = client.get("ubvf-ztfx", limit=800000)
    incident_df = pd.DataFrame.from_records(results)
    # List of columns to keep
    columns_to_keep = [
        'weather_1', 'collision_severity', 'type_of_collision', 
        'road_surface', 'road_cond_1', 'lighting', 
        'control_device', 'intersection', 'number_killed', 'number_injured','time_cat','mviw','ped_action','party1_type','party2_type','party1_move_pre_acc','party2_move_pre_acc'
    ]

    # Select only the specified columns
    incident_df_subset = incident_df[columns_to_keep]

    incident_df_subset['collision_severity'].value_counts()
    def remap_severity(severity):
        if severity == 'Injury (Complaint of Pain)':
            return 0  # Minor Injury
        elif severity == 'Injury (Other Visible)':
            return 1  # Moderate Injury
        else:
            return 2  # Severe Injury (including 'Injury (Severe)', 'Fatal', 'Medical')

    incident_df_subset['collision_severity_class'] = incident_df_subset['collision_severity'].apply(remap_severity)

    def map_weather_condition(condition):
        if condition in ['Clear', 'Cloudy', 'Raining', 'Not Stated']:
            return condition
        elif 'Fog' in condition:
            return 'Fog'
        elif condition in ['Wind', 'Snowing', 'Other', 'Other: NOT ON SCENE', 
                            'Other: Unknown', 'Other: NOT AT SCENE', 'Other: MISTING']:
            return 'Other'
        else:
            return 'Unknown'  # Fallback category

    incident_df_subset['weather_1_grouped'] = incident_df_subset['weather_1'].apply(map_weather_condition)

    # Convert the grouped categories to numerical values
    weather_grouped_mapping = {label: idx for idx, label in enumerate(incident_df_subset['weather_1_grouped'].unique())}
    incident_df_subset['weather_1_mapped'] = incident_df_subset['weather_1_grouped'].map(weather_grouped_mapping)

    intersection_mapping = {label: idx for idx, label in enumerate(incident_df_subset['intersection'].unique())}
    incident_df_subset['intersection_mapped'] = incident_df_subset['intersection'].map(intersection_mapping)

    road_cond_mapping = {label: idx for idx, label in enumerate(incident_df_subset['road_cond_1'].unique())}
    incident_df_subset['road_cond_1_mapped'] = incident_df_subset['road_cond_1'].map(road_cond_mapping)

    control_device_mapping = {label: idx for idx, label in enumerate(incident_df_subset['control_device'].unique())}
    incident_df_subset['control_device_mapped'] = incident_df_subset['control_device'].map(control_device_mapping)

    lighting_mapping = {label: idx for idx, label in enumerate(incident_df_subset['lighting'].unique())}
    incident_df_subset['lighting_mapped'] = incident_df_subset['lighting'].map(lighting_mapping)

    road_surface_mapping = {label: idx for idx, label in enumerate(incident_df_subset['road_surface'].unique())}
    incident_df_subset['road_surface_mapped'] = incident_df_subset['road_surface'].map(road_surface_mapping)

    type_of_collision_mapping = {label: idx for idx, label in enumerate(incident_df_subset['type_of_collision'].unique())}
    incident_df_subset['type_of_collision_mapped'] = incident_df_subset['type_of_collision'].map(type_of_collision_mapping)

    time_cat_mapping = {label: idx for idx, label in enumerate(incident_df_subset['time_cat'].unique())}
    incident_df_subset['time_cat_mapped'] = incident_df_subset['time_cat'].map(time_cat_mapping)

    party2_move_mapping = {label: idx for idx, label in enumerate(incident_df_subset['party2_move_pre_acc'].unique())}
    incident_df_subset['party2_move_pre_acc_mapped'] = incident_df_subset['party2_move_pre_acc'].map(party2_move_mapping)

    party1_move_mapping = {label: idx for idx, label in enumerate(incident_df_subset['party1_move_pre_acc'].unique())}
    incident_df_subset['party1_move_pre_acc_mapped'] = incident_df_subset['party1_move_pre_acc'].map(party1_move_mapping)

    ped_action_mapping = {label: idx for idx, label in enumerate(incident_df_subset['ped_action'].unique())}
    incident_df_subset['ped_action_mapped'] = incident_df_subset['ped_action'].map(ped_action_mapping)

    party1_type_mapping = {label: idx for idx, label in enumerate(incident_df_subset['party1_type'].unique())}
    incident_df_subset['party1_type_mapped'] = incident_df_subset['party1_type'].map(party1_type_mapping)

    party2_type_mapping = {label: idx for idx, label in enumerate(incident_df_subset['party2_type'].unique())}
    incident_df_subset['party2_type_mapped'] = incident_df_subset['party2_type'].map(party2_type_mapping)

    mviw_mapping = {label: idx for idx, label in enumerate(incident_df_subset['mviw'].unique())}
    incident_df_subset['mviw_mapped'] = incident_df_subset['mviw'].map(mviw_mapping)

    # List of mapped columns to keep
    mapped_columns = [
        'weather_1_mapped', 'intersection_mapped', 'road_cond_1_mapped', 
        'control_device_mapped', 'lighting_mapped', 'road_surface_mapped', 
        'type_of_collision_mapped', 'time_cat_mapped', 'party2_move_pre_acc_mapped', 
        'party1_move_pre_acc_mapped', 'ped_action_mapped', 'party1_type_mapped', 
        'party2_type_mapped', 'mviw_mapped','collision_severity_class'
    ]

    # Create a new DataFrame with only the mapped columns
    mapped_incident_df = incident_df_subset[mapped_columns]

    # Now, 'mapped_incident_df' contains only the mapped versions of your original categorical variables

    api = 'HepqrjAO5PJKyUmx.qU3vOWCYlgcF71nsB5JyDXDQqZPrIJmNK43gtPZzdXM5PSwhDAuotGM26t7ujy19'
    project = hopsworks.login(api_key_value = api)
    fs = project.get_feature_store()
    traffic_fg = fs.get_or_create_feature_group(
            name="sf_trarric_accidents",
            version=1,
            primary_key=['weather_1_mapped', 'intersection_mapped',
        'road_cond_1_mapped', 'control_device_mapped',
        'lighting_mapped', 'road_surface_mapped',
        'type_of_collision_mapped', 'time_cat_mapped',
        'party2_move_pre_acc_mapped', 'party1_move_pre_acc_mapped', 'ped_action_mapped',
        'party1_type_mapped', 'party2_type_mapped',
        'mviw_mapped','collision_severity_class'], 
            description="Incident dataset")
        
    traffic_fg.insert(mapped_incident_df)
    return "Data updated successfully"

def update_model():
    import hopsworks
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
    import seaborn as sns
    from matplotlib import pyplot
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib
    import os
    api = 'UtYWT9JBE4jbsOVW.dzfTExU7QMCzzR51EADTOZCXBzl0VmgB2y012yd8nFTG6v1VHgWazdx2a2SuJAY1'
    project = hopsworks.login(api_key_value = api)
    fs = project.get_feature_store()

    traffic_fg = fs.get_or_create_feature_group(
    name="sf_trarric_accidents",
    version=1,
    primary_key=['weather_1_mapped', 'intersection_mapped',
    'road_cond_1_mapped', 'control_device_mapped',
    'lighting_mapped', 'road_surface_mapped',
    'type_of_collision_mapped', 'time_cat_mapped',
    'party2_move_pre_acc_mapped', 'party1_move_pre_acc_mapped', 'ped_action_mapped',
    'party1_type_mapped', 'party2_type_mapped',
    'mviw_mapped','collision_severity_class'], 
    description="Incident dataset")
    query = traffic_fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="data_1",
                                    version=1,
                                    description="Read from SF traffic dataset",
                                    labels=["collision_severity_class"],
                                    query=query)
    
    X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)
    from sklearn.neural_network import MLPClassifier

    # Define the MLPClassifier
    # Here, we are using a simple architecture with two hidden layers, each with 100 neurons
    # You can experiment with different architectures and hyperparameters
    model = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=1000, random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    mlp_pred = model.predict(X_test)

    # Evaluate the model
    print("MLP Classification Report:\n", classification_report(y_test, mlp_pred))

    # Predicting Test Set
    # y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
    acc = accuracy_score(y_test, mlp_pred)
    prec = precision_score(y_test, mlp_pred, average='weighted')
    rec = recall_score(y_test, mlp_pred, average='weighted')
    f1 = f1_score(y_test, mlp_pred, average='weighted')

    results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1]],
                columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    print(results)


    # We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
    mr = project.get_model_registry()
    model_dir="sf_traffic_models"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)
    joblib.dump(model, model_dir + "/sf_traffic_model_1.pkl")

    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)


    # Ensure that the metrics are single numerical values
    sf_traffic_model = mr.python.create_model(
        name="sf_traffic_model_1", 
        metrics={
            "accuracy": results['Accuracy'].iloc[0],
            "Precision": results['Precision'].iloc[0],
            "Recall": results['Recall'].iloc[0],
            "F1_score": results['F1 Score'].iloc[0]
        },
        model_schema=model_schema,
        description="updated SF traffic model"
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    sf_traffic_model.save(model_dir)
    return results
# Add other mapping functions as necessary

def traffic_predict(weather, intersection, road_cond, control_device, 
                    lighting, road_surface, collision_type, time_cat, 
                    party2_move, party1_move, ped_action, party1_type, 
                    party2_type, mviw):
    # Apply mappings to each input
    weather_mapped = map_weather_condition(weather)
    intersection_mapped = map_intersection(intersection)
    road_cond_mapped = map_road_cond_1(road_cond)
    control_device_mapped = map_control_device(control_device)
    lighting_mapped = map_lighting(lighting)
    road_surface_mapped = map_road_surface(road_surface)
    collision_type_mapped = map_type_of_collision(collision_type)
    time_cat_mapped = map_time_cat(time_cat)
    party2_move_mapped = map_party2_move_pre_acc(party2_move)
    party1_move_mapped = map_party1_move_pre_acc(party1_move)
    ped_action_mapped = map_ped_action(ped_action)
    party1_type_mapped = map_party1_type(party1_type)
    party2_type_mapped = map_party2_type(party2_type)
    mviw_mapped = map_mviw(mviw)

    # Prepare DataFrame for prediction
    df = pd.DataFrame([[weather_mapped, intersection_mapped, road_cond_mapped, control_device_mapped, 
                        lighting_mapped, road_surface_mapped, collision_type_mapped, time_cat_mapped, 
                        party2_move_mapped, party1_move_mapped, ped_action_mapped, party1_type_mapped, 
                        party2_type_mapped, mviw_mapped]],
                      columns=['weather_1_mapped', 'intersection_mapped', 'road_cond_1_mapped', 
                               'control_device_mapped', 'lighting_mapped', 'road_surface_mapped', 
                               'type_of_collision_mapped', 'time_cat_mapped', 'party2_move_pre_acc_mapped', 
                               'party1_move_pre_acc_mapped', 'ped_action_mapped', 'party1_type_mapped', 
                               'party2_type_mapped', 'mviw_mapped'])

    # Predict with model
    prediction = model.predict(df)
    return prediction[0]


with gr.Blocks() as demo:
    with gr.Row():
        weather = gr.Dropdown(['Clear', 'Cloudy', 'Raining', 'Fog', 'Other', 'Not Stated', 'Unknown'], label="Weather Condition")
        intersection = gr.Dropdown(['Intersection <= 20ft', 'Midblock > 20ft', 'Intersection Rear End <= 150ft'], label="Intersection")
        road_cond = gr.Dropdown(['No Unusual Condition', 'Not Stated', 'Other', 'Construction or Repair Zone', 'Holes, Deep Ruts', 'Obstruction on Roadway', 'Loose Material on Roadway', 'Holes, Deep Rut', 'Reduced Roadway Width', 'Flooded'], label="Road Condition 1")
        control_device = gr.Dropdown(['Functioning', 'None', 'Not Stated', 'Not Functioning', 'Obscured'], label="Control Device")
        lighting = gr.Dropdown(['Daylight', 'Dark - Street Lights', 'Dusk - Dawn', 'Not Stated', 'Dark - No Street Lights', 'Dark - Street Lights Not Functioning'], label="Lighting")
        road_surface = gr.Dropdown(['Dry', 'Wet', 'Not Stated', 'Slippery', 'Snowy or Icy'], label="Road Surface")
        collision_type = gr.Dropdown(['Broadside', 'Vehicle/Pedestrian', 'Rear End', 'Sideswipe', 'Head-On', 'Other', 'Hit Object', 'Not Stated', 'Overturned'], label="Type of Collision")
        time_cat = gr.Dropdown(['2:01 pm to 6:00 pm', '10:01 am to 2:00 pm', '6:01 pm to 10:00 pm', '6:01 am to 10:00 am', '10:01 pm to 2:00 am', '2:01 am to 6:00 am'], label="Time Category")
        party2_move = gr.Dropdown(['Proceeding Straight', 'Stopped In Road', 'Not Stated', 'Making Left Turn', 'Parked', 'Other', 'Stopped', 'Making Right Turn', 'Slowing/Stopping', 'Entering Traffic', 'Backing', 'Changing Lanes', 'Parking Maneuver', 'Passing Other Vehicle', 'Making U Turn', 'Merging', 'Ran Off Road', 'Traveling Wrong Way', 'Crossed Into Opposing Lane', 'Other Unsafe Turning', 'Crossed Into Opposing Lane - Unplanned'], label="Party 2 Movement Pre-Accident")
        party1_move = gr.Dropdown(['Proceeding Straight', 'Making Left Turn', 'Making Right Turn', 'Changing Lanes', 'Other', 'Entering Traffic', 'Not Stated', 'Backing', 'Stopped In Road', 'Making U Turn', 'Parked', 'Slowing/Stopping', 'Passing Other Vehicle', 'Traveling Wrong Way', 'Stopped', 'Ran Off Road', 'Other Unsafe Turning', 'Parking Maneuver', 'Crossed Into Opposing Lane', 'Merging', 'Crossed Into Opposing Lane - Unplanned'], label="Party 1 Movement Pre-Accident")
        ped_action = gr.Dropdown(['No Pedestrian Involved', 'Crossing in Crosswalk at Intersection', 'Crossing Not in Crosswalk', 'In Road, Including Shoulder', 'Not in Road', 'Not Stated', 'Crossing in Crosswalk Not at Intersection', 'Approaching/Leaving School Bus'], label="Pedestrian Action")
        party1_type = gr.Dropdown(['Driver', 'Bicyclist', 'Pedestrian', 'Other', 'Parked Vehicle', 'Not Stated'], label="Party 1 Type")
        party2_type = gr.Dropdown(['Driver', 'Pedestrian', 'Bicyclist', 'Parked Vehicle', 'Other', 'Not Stated'], label="Party 2 Type")
        mviw = gr.Dropdown(['Other Motor Vehicle', 'Pedestrian', 'Bicycle', 'Fixed Object', 'Parked Motor Vehicle', 'Non-Collision', 'Not Stated', 'Other Object', 'Motor Vehicle on Other Roadway', 'Train', 'Animal'], label="MVIW")

    with gr.Row():
        predict_button = gr.Button("Predict Severity")
        update_data_button = gr.Button("Update Data")
        retrain_model_button = gr.Button("Retrain Model")

    output = gr.Textbox(label="Output")

    predict_button.click(fn=traffic_predict, inputs=[weather, intersection, road_cond, control_device, 
                        lighting, road_surface, collision_type, time_cat, 
                        party2_move, party1_move, ped_action, party1_type, 
                        party2_type, mviw], outputs=output)
    update_data_button.click(fn=update_data, inputs=[], outputs=output)
    retrain_model_button.click(fn=update_model, inputs=[], outputs=output)

demo.launch(share=True)

