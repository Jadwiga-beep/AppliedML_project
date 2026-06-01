# Applied ML project: Vegetable and fruit classifier 

## Description 
This project implements an SVM that is used as a baseline model and a CNN that classifies fruits and vegetables. This project trains 3 CNNs, which are trained on three color spaces (RGB, HSV, and greyscale). 

The CNN models are deployed using FastAPI, which allows users to upload an RGB image and receive predictions from all three models 

## Installation dependencies

1. Clone the repository

    ```bash
    git clone git@github.com:Jadwiga-beep/AppliedML\_project.git

    cd AppliedML_project
    ```

3. Create a virtual environment and install dependencies
    ```bash
     uv sync
    ```
5. Activate the virtual environment
    ```bash
     source .venv/bin/activate
    ```
## Launch API

Ensure that before running the API, run models.py so the models are saved 

1. Start API server
    ```bash
     uvicorn main:app --reload
    ```
3. Open API documentation
    ```bash
     http://127.0.0.1:8000/docs
    ```
