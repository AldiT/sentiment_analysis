
from typing import List, Dict, Union


from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{items_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {'item_id': item_id, "q": q}





if __name__ == "__main__":
    print("Running the api.py as a main file.")