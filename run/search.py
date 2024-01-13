import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import os
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

dir = "c:\\stock"
start_date = "2013-01-01"
end_date = "2024-01-07"
logger = None
machine_name = "stockAI2"
input_column = ["Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg", "TranAmnt"]
target_column = ['Close']
sequence_length = 60  # x축에 들어갈 데이터 길이
#end_date = "2024-01-07"

# 이전 모델 불러오기
model_target = load_model(f'{dir}\\machine\\{machine_name}.h5')
model_target.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss=tf.compat.v1.losses.mean_squared_error)

def check_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_list_to_csv(file_path, data):
    if os.path.exists(file_path):
        os.remove(file_path)
        
    with open(file_path, mode='w', newline='' , encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        
def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.FileHandler(f'{dir}\\log\\logfile_{name}.log')  # 로그 파일 이름 및 경로 지정
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger

def write_log(msg):
    logger.info(msg)
    print(msg)

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
        
def process_stock_data(stock):
    code, name, market = stock
    lastData = None
    data = [["Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg", "TranAmnt"]]
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
        lastData = row["Close"]
        data.append([row["Open"], row["High"], row["Low"], row["Close"], row["Change"], row["Volume"], row["5MvAvg"], row["20MvAvg"], row["60MvAvg"], row["112MvAvg"], row["224MvAvg"], row["UpperBand"], row["LowerBand"], row["20VolAvg"], row["60VolAvg"], row["Close"] * row["Volume"] ])
    if len(data) > 224:
        lastData2 = None
        target_data = pd.DataFrame(data[-224:],columns=["Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg", "TranAmnt"])
        predicted_values = inference(target_data)
        idx = 1
        for ret in predicted_values[0]:
            output_data.append([f"Day {idx:>6}", ret])
            lastData2 = ret
            idx += 1
        retType = "bed"
        if lastData < lastData2:
            if lastData * 1.1 < lastData2:
                retType = "verygood"
                print(f"*****************************************************  {name} ({code}) *****************************************************")
            else:
                retType = "good"
        else:
            retType = "bed"
        save_list_to_csv(f"{dir}\\result\\{retType}\\{code}.csv", output_data[-25:])
        write_log(f"({code}) was created!")
    else:
        write_log(f"({code}) was fault!")
    
if __name__ == "__main__":
    
    check_directory(dir)
    check_directory(f"{dir}\\log")
    check_directory(f"{dir}\\dataset")
    check_directory(f"{dir}\\result")
    check_directory(f"{dir}\\result\\verygood")
    check_directory(f"{dir}\\result\\good")
    check_directory(f"{dir}\\result\\bed")
    
    # logger 불러오기
    logger = setup_custom_logger('create_stock_dataset')

    stocks = [["Code", "Name", "Market"]]
    for index, row in fdr.StockListing('KRX').iterrows():
        if row["Market"] == 'KOSPI' or row["Market"] == 'KOSDAQ':
            stocks.append([row["Code"], row["Name"], row["Market"]])
    for stock in stocks[1:]:
        process_stock_data(stock)
    
