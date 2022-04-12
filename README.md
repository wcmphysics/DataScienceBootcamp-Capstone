# HDD Anomaly-Detection
This is a repository about predicting if a hard drive is going to fail or not within 30 days by machine learning algorithms and artificial neural networks. 

The prediction is based on a stacked model consists of XGBoost and an artificial neural network. The model is trained and verified for a specific hard-drive model using the data released by [BackBlaze](https://www.backblaze.com/b2/hard-drive-test-data.html#downloading-the-raw-hard-drive-test-data), and it scores at 84% for ROC-AUC. In addition, we deployed a trained XGBoost model as an API in [Heroku](https://felix-roc-capstone.herokuapp.com/) and a GUI in [streamlit.io](https://share.streamlit.io/felix-roc/hdd-anomaly-detection/streamlit_frontend), so users can easily use the model to check if their hard drives are going to fail. 


# Installation
One can set up a virtual environment and install the required standard packages simply by running

    make setup

After that, activate the environment

    source .venv/bin/activate

And run the following command to install the custom packages

    python setup.py install

# Training and Prediction
To train the model, one should first unzip the zipped csv file in the main, and then run:

    python -m src.train

To make prediction using the trained model, run:

    python -m src.predict

