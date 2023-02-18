import joblib
import pandas as pd
import streamlit as st

from flask import jsonify
from preprocess_data import DataProcessor

model = joblib.load("../models/kmeans.pt")
# model = joblib.load("../models/hdbscan.pt")
# model = joblib.load("../models/isolation_forest.pt")


def welcome():
    return "Welcome All"


def predict(df):
    """
    Process incoming data and output the model prediction
    ___
    description: Create a new prediction.
    tags:
        - predict
    requestBody:
        description: prediction to create.
        required: true
        content:
            application/json:
                schema:
                    $ref: '#/components/schemas/PredictRequest'
        responses:
            200:
                description: Empty.
            422:
                description: Validation Error
                content:
                    application/json:
                        schema:
                            $ref: "#/components/schemas/HTTPValidationError"


    """

    if df.shape[0] > 0:
        df = DataProcessor().process(df)
        df["PREDICT"] = model.predict(df.drop(["CLIENT_IP", "EVENT_ID"], axis=1))
        return df[["EVENT_ID", "PREDICT"]].to_dict()
    else:
        return "No input data provided"


def main():
    st.title("PT Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit HTTP Request Clusterization App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        print(df.head())
        if st.button("Predict"):
            result = predict(df)
            st.success("The output is {}".format(result))
    if st.button("About"):
        st.text("Built with Streamlit")


if __name__ == "__main__":
    main()

#%%
