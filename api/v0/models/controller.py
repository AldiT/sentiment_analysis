
from pydantic import JsonError
from utils import load_cache_pkl, save_cache_pkl, load_config
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi import UploadFile
from sentiment_analysis.preprocessing.run import remove_special_chars, remove_stopwords
from transformers import AutoTokenizer

from utils import f1_m
from run import config
from pathlib import Path
from typing import Dict, List

import tensorflow as tf
import numpy as np
import spacy
import json
import pickle as pkl
import yaml


class ModelController:
    """
    ModelController is the class controlling everything related to models.
    """
    
    def __init__(self, database_path: Path = Path("./model_database.yaml")):
        # TODO: Add a connection to a real database
        self.database = load_config(database_path)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.cached_model = None
        self.cached_model_id = None
        self.nlp = spacy.load("en_core_web_sm")


    def get_models(self) -> JSONResponse:
        """
        Method returns information about available models.

        Returns:
            Return a JsonReponse.
        """
        response = []
        for model in self.database["models"]:
            response.append({
                "id": model["id"],
                "name": model["name"],
                "version": model["version"]
            })
        
        return JSONResponse(content=response, status_code=200)


    def get_model(self, model_id: int) -> JSONResponse:
        """
        This method returns a model given
        Args:
            model_id: The id of a model.
            texts: A list of strings to predict sentiment on.

        Returns:
            Return a JsonReponse.
        """
        model_data = self._get_model_data(model_id)

        if model_data is None:
            return Response(content=f"Model with id: {model_id} not found!", status_code=404)
        else:
            model_path = self._get_cache_path(model_data['foldername'], model_data['filename'])
            return FileResponse(model_path, status_code=200)

    def get_prediction(self, model_id: int, text: str) -> JSONResponse:
        """
        This method runs sentiment prediction on a single text.
        Args:
            model_id: The id of a model.
            text: A string to predict sentiment on.

        Returns:
            Return a JsonReponse.
        """
        model_data = self._get_model_data(model_id)
        if model_data is None:
            return Response(content=f"Model with id: {model_id} not found!", status_code=404)
        
        model_path = self._get_cache_path(model_data['foldername'], model_data['filename'])

        if self.cached_model_id is None or self.cached_model_id != model_id:
            self._cached_model = tf.keras.models.load_model(model_path, custom_objects={"f1_m": f1_m})
            self._cached_model_id = model_id
        
        if not isinstance(text, str):
            return Response(content=f"Text is not in the proper format!", status_code=400)

        response = self._single_inference(text)

        return JSONResponse(response, status_code=200)
    
    def get_batch_prediction(self, model_id: int, texts: List[str]) -> JSONResponse:
        """
        This method runs sentiment prediction on multiple texts.
        Args:
            model_id: The id of a model.
            texts: A list of strings to predict sentiment on.

        Returns:
            Return a JsonReponse.
        """
        model_data = self._get_model_data(model_id)
        if model_data is None:
            return Response(content=f"Model with id: {model_id} not found!", status_code=404)
        
        model_path = self._get_cache_path(model_data['foldername'], model_data['filename'])

        if self.cached_model_id is None or self.cached_model_id != model_id:
            self._cached_model = tf.keras.models.load_model(model_path, custom_objects={"f1_m": f1_m})
            self._cached_model_id = model_id

        results = []
        for text in texts:
            if isinstance(text, str):
                results.append(self._single_inference(text))
            else:
                results.append({})
            
        return JSONResponse(content=results, status_code=200)
        

    def get_model_info(self, model_id: int) -> JSONResponse:
        """
        Method returns information about a model given it's id.
        Args:
            model_id: The id of a model.

        Returns:
            Return a JsonReponse.
        """
        model_data = self._get_model_data(model_id)
        if model_data is None:
            return Response(content=f"Model with id: {model_id} not found!", status_code=404)
        

        train_performance_file = self._get_cache_path(model_data["foldername"], model_data["train_history_filename"])
        test_performance_file = self._get_cache_path(model_data["foldername"], model_data["test_history_filename"])

        response = {
            "id": model_data["id"],
            "name": model_data["name"],
            "version": model_data["version"],
            "description": model_data["description"]
        }

        if train_performance_file.exists():
            with open(train_performance_file, "r") as handle:
                train_performance = json.load(handle)
                response["train_performance"] = train_performance

        if test_performance_file.exists():
            with open(test_performance_file, "rb") as handle:
                test_performance = pkl.load(handle)
                response["test_performance"] = {
                "test_loss": test_performance[0],
                "test_accuracy": test_performance[1],
                "test_f1_score": test_performance[2]
            }

        return JSONResponse(content=response, status_code=200)

    def preprocess(self, texts: List) -> List[str]:
        """
        Method runs the preprocessing functions one the passed texts.
        Args:
            texts: A list of strings.

        Returns:
            Return a list of strings.
        """
        texts = remove_special_chars(texts)
        texts = remove_stopwords(texts, model=self.nlp)
        return texts

    def _get_cache_path(self, foldername: str, filename) -> Path:
        """
        Method returns a path to the cache with the given parameters.
        Args:
            foldername: The name of the folder within the cache.
            filename: The name of the file in the above folder.

        Returns:
            Return a Path.
        """
        return Path(f"./cache/models/{foldername}/{filename}")

    def _get_model_data(self, model_id: int) -> Dict:
        """
        Method returns the data for a model with a given id.
        Args:
            id: The unique identifier of the model.

        Returns:
            Return a dictionary with information about the model with the given id. None if model is not found.
        """
        result = None

        for model in self.database["models"]:
            if model["id"] == model_id:
                result = model
                break
            
        return result
    
    def _single_inference(self, text) -> Dict:
        """
        Method to perform inference on a single text.
        Args:
            text: The text to predict sentiment for.

        Returns:
            Returns a dictionary with a label and the prediction score normalized between 0 and 1.
        """
        text = self.preprocess([text])[0]
        tokenized = self.tokenizer.encode(text)

        out = self._cached_model.predict([tokenized])[0]

        response = {
            "text_preprocessed": text,
            "label": self._get_label(np.argmax(out)),
            "score": str(np.max(out))
        }
        return response

    def _load_model_from_cache(self, folder: str, filename: str) -> tf.keras.Model:
        """
        Load the model from a local cache location.
        Args:
            folder: The folder within the cache where the model is located.
            filename: The name of the file in ./cache/folder .

        Returns:
            Returns a tensorflow.keras model.
        """
        model = tf.keras.models.load_model(f"./cache/{folder}/{filename}")
        return model

    def _get_label(self, position: int) -> Dict:
        """
        Return the label given the model maximum output position (argmax),
        Args:
            position: The position of the maximum logit in the prediction class, for sentiment analysis it can only be 0, 1 or 2.

        Returns:
            Returns a dictionary with a label and the prediction score normalized between 0 and 1.
        """
        if position == 0:
            return "NEUTRAL"
        elif position == 1:
            return "POSITIVE"
        elif position == 2:
            return "NEGATIVE"
