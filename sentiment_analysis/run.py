from sentiment_analysis.preprocessing.run import run as run_preprocessing
import logging

logger = logging.getLogger(__name__)

def run():
    run_preprocessing()
    #run_training()


if __name__ == "__main__":
    run()