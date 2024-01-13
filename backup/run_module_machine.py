import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging

# 입력과 타겟 시퀀스 생성
sequence_length = 224  # x축에 들어갈 데이터 길이
predict_days = 20  # 예측할 날짜
dir = "D:\\work"
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

if __name__ == "__main__":
    # 이전 모델 불러오기
    model_target = load_model(f'{dir}\\stock_model.keras')

    # 데이터 불러오기
    new_data = pd.read_csv(f'{dir}\\target_data.csv')

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


    # 데이터 추론
    last_data_input = data_input_normalized[-sequence_length:].reshape(1, sequence_length, len(input_column))
    predicted_values = model_target.predict(last_data_input)

    # 정규화 된 값을 다시 원래 스케일로 되돌림
    #predicted_values = predicted_values.reshape(predict_days, 1)
    predicted_values = scaler_target.inverse_transform(predicted_values)

    # 출력
    print("result: ", predicted_values)
    #logger = setup_custom_logger('run_module')
    #logger.info("result: " + str(predicted_values))
