from ast import Load
from pathlib import Path
from dotenv import load_dotenv

import logging
import yaml
import os
import sys

logging.basicConfig(filename='logs.log', format='%(asctime)s - %(filename)s - %(funcName)s - Line: %(lineno)d - %(levelname)s - %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p', filemode='w', level=logging.DEBUG)

logger = logging.getLogger(__name__)



def load_config():
    path = Path("./config.yaml")
    if not path.is_file():
        sys.exit(-1)

    with open("./config.yaml", "r") as handler:
        config = yaml.load(handler, Loader=yaml.FullLoader)
        logger.info("Config file read successfully!")

    return config

def set_environment_variables():
    load_dotenv()



if __name__ == "__main__":
    load_config()
    set_environment_variables()