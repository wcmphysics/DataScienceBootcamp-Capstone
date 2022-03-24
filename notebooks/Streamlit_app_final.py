#importing modules for connection and df
from tkinter import font
import requests

import pandas as pd
import streamlit as st 
import numpy as np


#importing preprocessing script
# import

#saving/loading the modell
import pickle
import warnings
warnings.filterwarnings('ignore')

# loading model and data
xgb_model = '../models/xgb_model.sav'
loaded_model = pickle.load(open(xgb_model, 'rb'))
X_test = pd.read_csv('../models/X_test.csv')
y_test = pd.read_csv('../models/y_test.csv')

# Defining Prediction Function
def predict_rating(loaded_model, df):
    
    y_pred = loaded_model.predict(df)
    
    return y_pred['Label'][0]



# Writing App Title and Description
st.title('How long will your hard drive last?')
st.write('This is a web app to predict if the HDD drive will fail in the next 30 days.\
        several features that you can see in the sidebar. Please click on the Predict button at the bottom to\
        see the prediction of the classification.')

#sidebar
st.sidebar.title('Fail or not to Fail!')
st.sidebar.image('../docs/Guardians_memory.jpg')
st.sidebar.header("Guardians of the Memory", )
st.sidebar.text('Felix, Chang Ming, Andreas & Daniela')



#upload a file
dataframe = None
uploaded_file = st.file_uploader("Choose a file", help= 'Drag your files here')
if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe.head(5))
        st.success('Your file was successfully uploaded!')
        st.balloons()

# df = preprocessed_data(x,y...)


# Predicting HDD failure
# executing preprocessing via pipeline
if st.button('Predict'):
    #
    y_pred = predict_rating(loaded_model, dataframe)
    
    st.write(' Based on feature values, the car star rating is '+ str(int(y_pred)))



if dataframe is not None:
    # url = 'http://0.0.0.0:9696/spam_detection_path/'
    HDD_failure = dataframe.to_json()
    r = requests.post(url, json=HDD_failure)
    print(r.text)

    Prediction_answer = r.json()
