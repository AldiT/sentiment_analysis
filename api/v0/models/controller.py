
from utils import load_cache_pkl, save_cache_pkl, load_config
from fastapi.responses import Response, FileResponse, JSONResponse
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
    
    def __init__(self, database_path: Path = Path("./model_database.yaml")):
        # TODO: Add a connection to a real database
        self.database = load_config(database_path)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.cached_model = None
        self.cached_model_id = None
        self.nlp = spacy.load("en_core_web_sm")


    def get_models(self):
        return JSONResponse(content=self.database["models"], status_code=200)


    def get_model(self, model_id: int):
        model_data = self._get_model_data(model_id)

        if model_data is None:
            return Response(content=f"Model with id: {model_id} not found!", status_code=404)
        else:
            model_path = self._get_cache_path(model_data['cache_folder'], model_data['filename'])
            return FileResponse(model_path, status_code=200)

    def get_prediction(self, model_id: int, text: str):
        model_data = self._get_model_data(model_id)
        if model_data is None:
            return Response(content=f"Model with id: {model_id} not found!", status_code=404)
        
        model_path = self._get_cache_path(model_data['cache_folder'], model_data['filename'])

        if self.cached_model_id is None or self.cached_model_id != model_id:
            self._cached_model = tf.keras.models.load_model(model_path, custom_objects={"f1_m": f1_m})
            self._cached_model_id = model_id
        
        if not isinstance(text, str):
            return Response(content=f"Text is not in the proper format!", status_code=400)

        text = self.preprocess([text])[0]
        tokenized = self.tokenizer.encode(text)

        out = self._cached_model.predict([tokenized])[0]

        response = {
            "text_preprocessed": text,
            "label": self._get_label(np.argmax(out)),
            "score": str(np.max(out))
        }

        return JSONResponse(response, status_code=200)

    def preprocess(self, texts: List):

        texts = remove_special_chars(texts)
        texts = remove_stopwords(texts, model=self.nlp)
        return texts

    def _get_cache_path(self, foldername: str, filename):
        return Path(f"./cache/models/{foldername}/{filename}")

    def _get_model_data(self, model_id: int):
        result = None

        for model in self.database["models"]:
            if model["id"] == model_id:
                result = model
                break
            
        return result

    def _load_model_from_cache(self, folder: str, filename: str):
        model = tf.keras.models.load_model(f"./cache/{folder}/{filename}")
        return model

    def _get_label(self, position: int):
        if position == 0:
            return "NEUTRAL"
        elif position == 1:
            return "POSITIVE"
        elif position == 2:
            return "NEGATIVE"
