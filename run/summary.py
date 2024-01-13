import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging
import os
import csv
import tensorflow as tf

dir = "c:\\stock"
machine_name = "stockAI2"
logger = None

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.FileHandler(f'{dir}\\log\\logfile_{name}.log')  # 로그 파일 이름 및 경로 지정
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger
    
def write_log(msg):
    global logger
    if logger == None:
        logger = setup_custom_logger('run_stock_ai')
    logger.info(msg)
    print(msg)
  

if __name__ == "__main__":
    
    # 이전 모델 불러오기
    model_target = load_model(f'{dir}\\machine\\{machine_name}.h5')
    model_target.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss=tf.compat.v1.losses.mean_squared_error)
    model_target.summary()
    #model_target.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss=tf.compat.v1.losses.mean_squared_error)

