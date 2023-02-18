import joblib
import pandas as pd
import json

from flask import Flask, request, jsonify
from flasgger import Swagger
from preprocess_data import DataProcessor


app = Flask(__name__)
swagger = Swagger(
    app,
    template={
        "swagger": "3.0",
        "openapi": "3.0.0",
        "info": {
            "title": "predict",
            "version": "0.0.1",
        },
        "paths": {
            "/predict": {
                "post": {
                    "summary": "Predict",
                    "operationId": "predict_predict_post",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "title": "Requests",
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/PredictRequest"},
                                }
                            }
                        },
                        "required": True,
                    },
                    "responses": {
                        "200": {"description": "Successful Response", "content": {"application/json": {"schema": {}}}},
                        "422": {
                            "description": "Validation Error",
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}
                            },
                        },
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "HTTPValidationError": {
                    "title": "HTTPValidationError",
                    "type": "object",
                    "properties": {
                        "detail": {
                            "title": "Detail",
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/ValidationError"},
                        }
                    },
                },
                "PredictRequest": {
                    "title": "PredictRequest",
                    "required": ["data"],
                    "type": "object",
                    "properties": {"data": {"title": "JSON serialized string", "type": "string"}},
                },
                "ValidationError": {
                    "title": "ValidationError",
                    "required": ["loc", "msg", "type"],
                    "type": "object",
                    "properties": {
                        "loc": {
                            "title": "Location",
                            "type": "array",
                            "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                        },
                        "msg": {"title": "Message", "type": "string"},
                        "type": {"title": "Error Type", "type": "string"},
                    },
                },
            }
        },
    },
)

model = joblib.load("../models/kmeans.pt")
# model = joblib.load("../models/hdbscan.pt")
# model = joblib.load("../models/isolation_forest.pt")


@app.route("/")
def welcome():
    return "Welcome All"


@app.route("/predict", methods=["POST"])
def predict():
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
    r = request.get_json()
    r = [json.loads(i["data"]) for i in r]

    df = pd.json_normalize(r)

    if df.shape[0] > 0:
        df = DataProcessor().process(df)
        df["PREDICT"] = model.predict(df.drop(["CLIENT_IP", "EVENT_ID"], axis=1))
        return jsonify(df[["EVENT_ID", "PREDICT"]].to_dict()), 200
    else:
        return jsonify({"message": "No input data provided"}), 400


if __name__ == "__main__":
    app.run()
