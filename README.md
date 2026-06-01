# AppliedML_project
# Vegetable and fruit classifier 

## Description 
This project implements a SVM that is used as a baseline model and a CNN that classify fruits and vegetables. This project trains 3 CNNs which are trained on  three color spaces images (RGB, HSV and greyscale). 

The CNN models are deployed using a FastAPI whih allows the users to upload a RGB image and recive predictions from all threee models 

## Installation dependencies

1. Clone repository 
git clone <git@github.com:Jadwiga-beep/AppliedML\_project.git>
cd AppliedML_project

2. Create and activate virtual enviroment 
uv sync
source .venv/bin/activate

## Launch API

Ensure that before running the API models.py has run so the models are saved 

1. Start API server 
uvicorn main:app --reload

2. Open API documentation 
http://127.0.0.1:8000/docs