#!/bin/bash

curl \
  -u token:$DATABRICKS_TOKEN \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
        "columns": ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],
        "data": [[6.5,0.28,0.35,15.4,0.042,55,195,0.9978,3.23,0.5,9.6]]
    }
  }' \
  http://0.0.0.0:1234/invocations