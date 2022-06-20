from utils import load_config, set_environment_variables
import os

config = load_config()
set_environment_variables()




if __name__ == "__main__":
    print(os.environ["TEST_VAR"]) 