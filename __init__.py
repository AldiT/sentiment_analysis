import logging

logging.basicConfig(filename='logs.log', format='%(asctime)s - %(filename)s - %(module)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p', filemode='w', level=logging.DEBUG)