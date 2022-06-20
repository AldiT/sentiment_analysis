# Sentiment analysis system 
[![Generic badge](https://img.shields.io/badge/python-3.8-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/docker-20.10.8-blue.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/docker-21.2.4-yellow.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/docker_build-passed-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/stability-experimental-orange.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/version-0.1-aquamarine.svg)](https://shields.io/)

### Detailed code documentation can be found on: <a href="http://aldit-sentiment-technical-docs.s3-website.eu-central-1.amazonaws.com"> this link</a>

## What is in this repository?
- ```api/``` This is where the api code is located, currently at version (folder) ```v0```.
- ```sentiment_analysis``` This is where the sentiment analysis logic is, from preprocessing to training.
- ```config.yaml``` <br/> The config file contains configuration for the whole application. It is meant to be a central point to steer the application instead of modifying the code directly. Functionality for hot reloading is not yet implemented.
- ```dataset_database.yaml``` and ```model_database.yaml``` <br/>
    Since time was short and there was not time to configure a proper database (Postgres or else) I tried to simulate one using yaml files for the purpose of this prototype.
- ```main.py``` This is the main entrypoint for the fastapi framework.
- ```utils.py``` The utils.py file contains helper functions that are used on various parts of the codebase.


For more details on the code you should redirect to the <a href="http://aldit-sentiment-technical-docs.s3-website.eu-central-1.amazonaws.com"> code documentation</a>. The code documentation is not complete nor is it final, it is simply meant to give a easy way to navigate through the code. Here all arguments and return objects are explained for almost all the methods present.

# The Model
The model that is trained in the notebook located on ```./sentiment_analysis/train/model_prototype_lstm.ipynb```, is a Bidirectional LSTM. The reason for choosing this model is simple: it is relatively light weight and it can train reasonably fast. It is not state of the art but it serves well the purpose of a prototype.
<br/><br/>
The model was trained on the Google Colab platform on TPUs (Tensor Processing Units) and it takes roughly 190 seconds per epoch to train.