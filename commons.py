import logging
import os
import random


formatter = logging.Formatter('%(asctime)s; %(levelname)s; %(message)s')


def clear_log_files():
    for file_name in os.listdir('./logs/'):
        open('./' + file_name, 'w').close()


def setup_logger(name, file, level=logging.INFO):
    handler = logging.FileHandler(file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def create_env_map(number_of_columns, number_of_rows, amount_of_walls=.0):
    # TODO: How to ensure that there is no unpassable border?
    first_line = '+' + (number_of_columns - 1) * '--' + '-+'
    env_map = [first_line]
    for r in range(number_of_rows):
        str = '|'
        for c in range(number_of_columns - 1):    # -1 bc last one isn't random
            if random.random() < amount_of_walls:  
                str += ' |'
            else:
                str += ' :'
        str += ' |'
        env_map.append(str)
    env_map.append(first_line)
    return env_map