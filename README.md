# Applied ML project: Fruits & Vegetable Classifier

![Image of an Apple in RGB, HSV, and Grayscale](images/rgb,hsv,gray,example.png)

## Description
This project implements a CNN that classifies fruits and vegetables and an SVM that is used as a baseline model. This project trains 3 CNNs, which are trained on three color spaces (RGB, HSV, and Grayscale). 

The CNN models are deployed using FastAPI, which allows users to upload an RGB image and receive predictions from all three models.

## Prerequisites 

- Docker installed

## Code Structure

```
data/               Folder with the images of our dataset
    fruits/         Folder containing all images of fruit - 5 classes
    vegetables/     Folder containing all images of vegetables - 5 classes
images/             Folder with code-generated images and plots
models/             Folder with pre-trained models used by the API and list of class names as Json file
api.py              File building the API
CNN.py              File with the CNN model
Dockerfile          File that copies the relevant files
evaluation.py       File with evaluation functiosn for the CNN models
main.py             Main file which loads and preprocesses the data, trains, saves, and evaluates the models
plots.py            File generating plots for initial data analysis
preprocessing.py    File with preprocessing functions
SVM.py              File with the SVM model
train.py            File with training and saving functions
```

## Opening the code

Clone the repository

```bash
    git clone git@github.com:Jadwiga-beep/AppliedML_project.git
    cd AppliedML_project
```

## Training the Model

The repository already includes pretrained model files in `models/`.

If you want to retrain the models, run:
```bash
    python3 main.py
```

This file includes preprocessing of the data, training the models, and evaluating the models on the test set.
Additionally, it trains and evaluates the SVM baseline model.


## Activate Virtual Environemnt

1. Clone the repository

```bash
    git clone git@github.com:Jadwiga-beep/AppliedML_project.git
    cd AppliedML_project
```

2. Create a virtual environment and install dependencies

```bash
    uv sync
```

3. Activate the virtual environment

```bash
    source .venv/bin/activate
```

4. In order to train the model, run `main.py`. 

```bash
    python3 main.py
```


## Loadding the API

5.1 Run the uvicorn comand in a terminal to initate the API.

```
    uvicorn api:app --reload
```

5.2 Or alternativley, use the docker pipeline.

```bash
    docker compose up --build
```

### 6.1 Through a browser

Open the interactive docs and use the `/predict` endpoint to upload an image directly:

```
    http://127.0.0.1:8000/docs
```

### 6.2 Using `check.sh`

`check.sh` is a bash script that takes an image URL and first downloads it, then sends it to the API, and prints the result. It additionally uses `|jq` to make the print more readable which can be installed with `sudo apt install jq` if needed. Make it executable once:

```bash
chmod +x check.sh
```

Then pass it an image URL in quotations:

```bash
./check.sh "image-address"
```

### 6.3 Using `curl` directly

To send a local image file to the API yourself:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@path/to/image.jpg"
```
