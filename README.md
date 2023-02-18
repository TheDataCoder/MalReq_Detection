# 

# Project structure
```
├── README.md
├── Dockerfile
├── data
│   ├── part_10.csv
│   └── part_test.csv
├── models
│   ├── dbscan.pt
│   ├── hdbscan.pt
│   ├── isolation_forest.pt
│   ├── kmeans.pt
│   └── scaler.pt
├── notebooks
│   └── solution_pt.ipynb
├── requirements.txt
└── src
    ├── __pycache__
    │   ├── preprocess_data.cpython-38.pyc
    │   └── utils.cpython-38.pyc
    ├── openapi.json
    ├── predict_flasgger.py
    ├── predict_streamlit.py
    ├── preprocess_data.py
    └── utils.py
```

## Installation process

- [ ] Please refer to requirements.txt for the required packages or run a docker container: either use pip or anaconda 

## Solution information

- [ ] I've used KMeans, DBScan, HDBScan and IsolationRandomForest as main algorithms for segmentation. 
- [ ] IsolationRandomForest provides general solution - "suspicious requests" predictions, while other algorithms segment the data into number of clusters.

## Run options

- [ ] Run streamlit app and upload csv file to get the predictions: streamlit run predict_streamlit.py
- [ ] Send POST request to the flask app: predict_flasgger.py
- [ ] Can be deployed with a docker via <code>docker build -t pt_project_api .</code>

## Todos:
    
- [ ] Add unit tests
- [ ] Add multiple model inference options

## Licence

Licenced by Apache 2.0 licence;