import os
import modal
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("ID2223"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
    model_version = 1
    api = 'UtYWT9JBE4jbsOVW.dzfTExU7QMCzzR51EADTOZCXBzl0VmgB2y012yd8nFTG6v1VHgWazdx2a2SuJAY1'
    project = hopsworks.login(api_key_value = api)
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("sf_traffic_model_1", version=model_version)
    model_dir = model.download()
    print("model_dir:",model_dir)
    model = joblib.load(model_dir + "/sf_traffic_model_1.pkl")
    feature_view = fs.get_feature_view(name="data_1", version=1)
    batch_data = feature_view.get_batch_data()
    y_pred = model.predict(batch_data)
    count_0 = (y_pred == 0).sum()
    count_1 = (y_pred == 1).sum()
    count_2 = (y_pred == 2).sum()
    print("number of severity of 0:",count_0)
    print("number of severity of 1:",count_1)
    print("number of severity of 2:",count_2)
    print("Traffic accident severity average:: ", y_pred.mean())

    #print(y_pred)


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f.remote()

