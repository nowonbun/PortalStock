import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
import logging

# 입력과 타겟 시퀀스 생성
sequence_length = 224  # x축에 들어갈 데이터 길이
predict_days = 20  # 예측할 날짜
#dir = "D:\\stock\\dataset"
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

    return logger

# LSTM 모델 생성
#model_target = Sequential([
#    LSTM(50, return_sequences=True, input_shape=(sequence_length, len(input_column))),  # 입력 데이터는 Open, High, Low, Close, Volume이므로 5개 열
#    LSTM(50, return_sequences=False),
#    Dense(predict_days)
#])
if __name__ == "__main__":
    # logger 불러오기
    logger = setup_custom_logger('create_module')
    
    # LSTM 모델 생성
    model_target = Sequential()
    # 첫 번째 LSTM 레이어
    model_target.add(LSTM((len(input_column)*10) + predict_days + 20, return_sequences=True, input_shape=(sequence_length, len(input_column))))
    # 두 번째 LSTM 레이어
    model_target.add(LSTM((len(input_column)*10) + predict_days + 10, return_sequences=False))
    # 추가적인 Dense 레이어 (옵션)
    model_target.add(Dense(sequence_length * len(input_column)))
    for i in range(len(input_column)):
        model_target.add(Dense(((len(input_column)-i)*10) + predict_days))  # 더 많은 노드를 갖는 추가적인 Dense 레이어도 추가할 수 있습니다.
    # 출력 레이어
    model_target.add(Dense(predict_days))
    # 모델 컴파일
    model_target.compile(optimizer='adam', loss='mean_squared_error')
    model_target.save(f'{dir}\\stock_model_test_20230108.h5')

    # 새로운 데이터 불러오기
    codelist = pd.read_csv(f'{dir}\\dataset\\list.csv')
    index = 0
    dataset_list = []
    for code in codelist["Code"]:
        try:
    # for code in codelist["Code"][2:]:
    #     try:
            print(code + " is load!!")
            logger.info(code + " is load!!")
            print("index - " + str(index))
            logger.info("index - " + str(index))
            
            index = index+1
            
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

            #dataset_list.append((x_input, y_target))
            # 모델 학습 (1번 선행학습)
            model_target.fit(x_input, y_target, batch_size=64, epochs=1)
            # 모델 저장
            model_target.save(f'{dir}\\stock_model_test_20230108.h5')
            
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_target.save(f'{dir}\\stock_model_{current_time}.keras')
            
        except FileNotFoundError:
            # 파일을 찾을 수 없을 때 실행할 코드
            print("not file " + code)
            
    # 전체 데이터셋 합치기
    #x_combined = np.concatenate([x[0] for x in dataset_list])
    #y_combined = np.concatenate([y[1] for y in dataset_list])
        
    # 모델 학습 (1번 선행학습)
    #model_target.fit(x_combined, y_combined, batch_size=64, epochs=1)
    # 모델 저장
    #model_target.save(f'{dir}\\stock_model_test_20230108.h5')



