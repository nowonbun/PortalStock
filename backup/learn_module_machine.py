import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import logging

# GPU 디바이스 목록 출력
print("사용 가능한 GPU 목록:", tf.config.list_physical_devices('GPU'))

# 입력과 타겟 시퀀스 생성
sequence_length = 224  # x축에 들어갈 데이터 길이
predict_days = 20  # 예측할 날짜
dir = "D:\\work\\dataset"
input_column = ["Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg"]
target_column = ['Close']

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

    handler = logging.FileHandler(f'{dir}\\logfile_{name}.log')  # 로그 파일 이름 및 경로 지정
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger

# 리스트 데이터 불러오기
codelist = pd.read_csv(f'{dir}\\list.csv')
index = 0
f_index = 0
b_index = index
r_count = 15

if __name__ == "__main__":
    # logger 불러오기
    logger = setup_custom_logger('learn_module')
    # 이전 모델 불러오기
    model_target = load_model(f'{dir}\\stock_model.keras')

    for i in range(r_count):
        for code in codelist["Code"][b_index:]:
            try:
        # for code in codelist["Code"][2:]:
        #     try:
                print("r_count - " + str(i))
                logger.info("r_count - " + str(i))
                print("index - " + str(index))
                logger.info("index - " + str(index))
                index = index+1
                print(code + " is load!!")
                logger.info(code + " is load!!")
                # 새로운 데이터 불러오기
                new_data = pd.read_csv(f'{dir}\\{code}.csv')
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
                    model_target.fit(x_input, y_target, batch_size=64, epochs=i+1)
                    
                    # 모델 저장
                    model_target.save(f'{dir}\\stock_model.h5')
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_target.save(f'{dir}\\backup\\stock_model_{current_time}.keras')
            except FileNotFoundError:
                # 파일을 찾을 수 없을 때 실행할 코드
                print("not file " + code)


