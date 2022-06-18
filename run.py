from utils import load_config, set_environment_variables
import os


if __name__ == "__main__":
    config = load_config()
    set_environment_variables()


    print(os.environ["TEST_VAR"])