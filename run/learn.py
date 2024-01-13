import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import logging
import FinanceDataReader as fdr
import os
import csv

# GPU 디바이스 목록 출력
print("사용 가능한 GPU 목록:", tf.config.list_physical_devices('GPU'))

# 입력과 타겟 시퀀스 생성
sequence_length = 60  # x축에 들어갈 데이터 길이
predict_days = 1  # 예측할 날짜
dir = "c:\\stock"
input_column = ["Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg", "TranAmnt"]
target_column = ['Close']
machine_name = "stockAI2"
logger = None
# 리스트 데이터 불러오기
index = 0
c_index = 10000000000000000
f_index = 0
b_index = index
r_count = 15
# 0.1이 보통으로, 낮으면 수렴율이 낮아진다.
# 초기에는 0.1로 훈련이 반복될 수록 수렴율을 낮춘다.
epochs = 2
learning_rate = 0.1 / epochs


def create_sequences(data_input, data_target, seq_length, predict_days):
    x = []
    y = []
    for i in range(len(data_input) - seq_length - predict_days + 1):
        x.append(data_input[i:i+seq_length])
        y.append(data_target[i+seq_length:i+seq_length+predict_days])
    return np.array(x), np.array(y)

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
        logger = setup_custom_logger('learn_stock_ai')
    logger.info(msg)
    print(msg)

def save_list_to_csv(file_path, data):
    if os.path.exists(file_path):
        os.remove(file_path)
        
    with open(file_path, mode='w', newline='' , encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

if __name__ == "__main__":
    # logger 불러오기
    logger = setup_custom_logger('learn_stock_ai')
    
    # stocks = [["Code", "Name", "Market"]]
    # for index, row in fdr.StockListing('KRX').iterrows():
    #     if row["Market"] == 'KOSPI' or row["Market"] == 'KOSDAQ':
    #         stocks.append([row["Code"], row["Name"], row["Market"]])
    # save_list_to_csv("%s\\dataset\\list.csv" % dir, stocks)
    # write_log("list was created")
    
    codelist = pd.read_csv(f'{dir}\\dataset\\list.csv')
    
    # 이전 모델 불러오기
    model_target = load_model(f'{dir}\\machine\\{machine_name}.h5')
    # model_target.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss=tf.compat.v1.losses.mean_squared_error)
    # write_log("optimizer is SGD!!")
    # # 모델 학습 (심화학습)
    # model_target.fit(x_input, y_target, batch_size=64, epochs=3)

    # 하이퍼 파라미터 설정
    adam_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,    # 학습률 설정
        beta_1=0.9,                     # 베타1 설정
        beta_2=0.999,                   # 베타2 설정
        epsilon=1e-07                   # 엡실론 설정
    )
    model_target.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss=tf.compat.v1.losses.mean_squared_error)
    write_log("optimizer is AdamOptimizer!!") 

    #for i in range(5,r_count,1):
    for code in codelist["Code"][b_index:]:
        if index >= c_index:
            break
        try:
            #write_log("r_count - " + str(i))
            write_log("index - " + str(index))
            write_log(code + " is load!!")
            index = index + 1
            
            # 새로운 데이터 불러오기
            new_data = pd.read_csv(f'{dir}\\dataset\\{code}.csv')
            # 필요한 열 선택 (Open, High, Low, Close, Change, Volume, 5MvAvg, 20MvAvg, 60MvAvg, 112MvAvg, 224MvAvg, UpperBand, LowerBand, 20VolAvg, 60VolAvg)
            new_data_input = new_data[input_column]
            # target을 두개로 잡으면 아래 Dense에서 에러가 발생한다. predict_days가 두개라서??
            # new_data_target = new_data[['Close', 'Volume']]
            new_data_target = new_data[target_column]

            # 데이터 정규화
            scaler_input = MinMaxScaler()
            scaler_target = MinMaxScaler()

            # 입력과 타겟 데이터 정규화
            data_input_normalized = scaler_input.fit_transform(new_data_input)
            data_target_normalized = scaler_target.fit_transform(new_data_target)

            x_input, y_target = create_sequences(data_input_normalized, data_target_normalized, sequence_length, predict_days)

            # GPU를 사용하여 모델 학습
            with tf.device('/GPU:0'):  # 첫 번째 GPU 사용
                # 모델 학습 (심화학습)
                model_target.fit(x_input, y_target, batch_size=64, epochs=epochs)

                aa = scaler_target.inverse_transform(model_target.predict(x_input[-1:]))
                bb = scaler_target.inverse_transform(y_target[-1:][0])
                print(f"******** predict : real = {aa} : {bb}")
                
                model_target.evaluate(x_input[-1:], y_target[-1:])
                
                # 모델 저장
                model_target.save(f'{dir}\\machine\\{machine_name}.h5')
                #current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                #model_target.save(f'{dir}\\machine\\backup\\{machine_name}_{current_time}.h5')
        except FileNotFoundError:
            # 파일을 찾을 수 없을 때 실행할 코드
            print("not file " + code)
