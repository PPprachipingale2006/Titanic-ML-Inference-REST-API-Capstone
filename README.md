# Titanic ML Inference REST API

## Project Overview

This project deploys a trained Titanic Survival Classification
model through a REST API using FastAPI.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- Docker
- Pytest

## Project Structure

- app/main.py: FastAPI server
- app/schemas.py: Input and response validation
- models/champion_model.joblib: Trained ML model
- models/metadata.json: Model metadata
- tests/test_api.py: Automated tests
- requirements.txt: Dependencies
- Dockerfile: Container configuration

## Model Details

Problem: Binary Classification

Dataset: Titanic

Target: Survived

Classes:
- 0: Not Survived
- 1: Survived

## API Endpoints

GET /

Health check endpoint.

POST /predict

Returns a survival prediction and class probabilities.

## Run Locally

Install requirements:

pip install -r requirements.txt

Run API:

uvicorn app.main:app --reload

Swagger documentation:

http://127.0.0.1:8000/docs

## Run Tests

pytest -v

## Docker

Build:

docker build -t titanic-ml-api .

Run:

docker run -p 8000:8000 titanic-ml-api

## Architecture

Jupyter Notebook -> Trained Model -> Joblib Export ->
FastAPI -> JSON Prediction -> Docker Container

## Author

Prachi Kiran Pingale

B.Sc. Computer Science