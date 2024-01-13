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

#target_code = "001060"
target_code = "093520"
# 입력과 타겟 시퀀스 생성
sequence_length = 60  # x축에 들어갈 데이터 길이
dir = "c:\\stock"
input_column = ["Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg", "TranAmnt"]
target_column = ['Close']
machine_name = "stockAI2"
start_date = "2020-01-01"
#end_date = "2023-10-17"
#end_date = "2023-11-10"
end_date = "2024-01-09"
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
    
def save_list_to_csv(file_path, data):
    if os.path.exists(file_path):
        os.remove(file_path)
        
    with open(file_path, mode='w', newline='' , encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def write_log(msg):
    global logger
    if logger == None:
        logger = setup_custom_logger('run_stock_ai')
    logger.info(msg)
    print(msg)
    
def process_stock_data(code):
    data = [input_column]
    stock_data = fdr.DataReader(code, start_date, end_date)
    stock_data['5MvAvg'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['20MvAvg'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['60MvAvg'] = stock_data['Close'].rolling(window=60).mean()
    stock_data['112MvAvg'] = stock_data['Close'].rolling(window=112).mean()
    stock_data['224MvAvg'] = stock_data['Close'].rolling(window=224).mean()
    stock_data['448MvAvg'] = stock_data['Close'].rolling(window=448).mean()
    
    stock_data['60StdDev'] = stock_data['Close'].rolling(window=60).std()
    
    stock_data['UpperBand'] = stock_data['60MvAvg'] + (stock_data['60StdDev'] * 2)
    stock_data['LowerBand'] = stock_data['60MvAvg'] - (stock_data['60StdDev'] * 2)
    
    stock_data['20VolAvg'] = stock_data['Volume'].rolling(window=20).mean()
    stock_data['60VolAvg'] = stock_data['Volume'].rolling(window=60).mean()

    output_data = []
    for index, row in stock_data.iterrows():
        if np.isnan(row["Open"]) == True:
            continue
        if np.isnan(row["High"]) == True:
            continue
        if np.isnan(row["Low"]) == True:
            continue
        if np.isnan(row["Close"]) == True:
            continue
        if np.isnan(row["Change"]) == True:
            continue
        if np.isnan(row["Volume"]) == True:
            continue
        if np.isnan(row["5MvAvg"]) == True:
            continue
        if np.isnan(row["20MvAvg"]) == True:
            continue
        if np.isnan(row["60MvAvg"]) == True:
            continue
        if np.isnan(row["112MvAvg"]) == True:
            continue
        if np.isnan(row["224MvAvg"]) == True:
            continue
        if np.isnan(row["UpperBand"]) == True:
            continue
        if np.isnan(row["LowerBand"]) == True:
            continue
        if np.isnan(row["20VolAvg"]) == True:
            continue
        if np.isnan(row["60VolAvg"]) == True:
            continue
        output_data.append([index.strftime('%Y-%m-%d'), str(row["Close"])])
        data.append([row["Open"], row["High"], row["Low"], row["Close"], row["Change"], row["Volume"], row["5MvAvg"], row["20MvAvg"], row["60MvAvg"], row["112MvAvg"], row["224MvAvg"], row["UpperBand"], row["LowerBand"], row["20VolAvg"], row["60VolAvg"], row["Close"] * row["Volume"] ])
    if len(data) > 224:
        target_data = pd.DataFrame(data[-224:],columns=["Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg", "TranAmnt"])
        predicted_values = inference(target_data)
        idx = 1
        for ret in predicted_values[0]:
            output_data.append([f"Day {idx:>6}", ret])
            idx += 1
        
        for ret in output_data[-25:]:
            print(f"{ret[0]:<20} \t {ret[1]}")
    else:
        write_log(f"({code}) was fault!")

def inference(data):
    # 데이터 불러오기
    new_data = data
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
    return predicted_values

if __name__ == "__main__":
    
    # 이전 모델 불러오기
    model_target = load_model(f'{dir}\\machine\\{machine_name}.h5')
    #model_target.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss=tf.compat.v1.losses.mean_squared_error)

    process_stock_data(target_code)
