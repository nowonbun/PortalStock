# 이건 create_stock_dataset과 create_module_machine을 합친 것과 같다.
# 업뎃이 이루어지면 필히 위 소스도 같이 수정해 놓자
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import os
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


dir = "c:\\stock"
start_date = "2013-01-01"
end_date = "2024-01-07"
logger = None
input_column = ["Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg", "TranAmnt"]
target_column = ['Close']
# 입력과 타겟 시퀀스 생성
sequence_length = 60  # x축에 들어갈 데이터 길이
predict_days = 1  # 예측할 날짜
machine_name = "stockAI2"
# 현재는 약 1억개의 파라미터를 가지고 있다.
# predict_days=33
# weight = 1.2
# sequence_length = 224
# weight = 1.5
# predict_days = 2
# weight = 2.2
weight = 2.2
weight2 = 2

def check_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# def save_list_to_csv(file_path, data):
#     if os.path.exists(file_path):
#         os.remove(file_path)
        
#     with open(file_path, mode='w', newline='' , encoding='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerows(data)
        
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
        
# def process_stock_data(stock):
#     code, name, market = stock
#     data = [input_column]
#     stock_data = fdr.DataReader(code, start_date, end_date)
#     stock_data['5MvAvg'] = stock_data['Close'].rolling(window=5).mean()
#     stock_data['20MvAvg'] = stock_data['Close'].rolling(window=20).mean()
#     stock_data['60MvAvg'] = stock_data['Close'].rolling(window=60).mean()
#     stock_data['112MvAvg'] = stock_data['Close'].rolling(window=112).mean()
#     stock_data['224MvAvg'] = stock_data['Close'].rolling(window=224).mean()
#     stock_data['448MvAvg'] = stock_data['Close'].rolling(window=448).mean()
    
#     stock_data['60StdDev'] = stock_data['Close'].rolling(window=60).std()
    
#     stock_data['UpperBand'] = stock_data['60MvAvg'] + (stock_data['60StdDev'] * 2)
#     stock_data['LowerBand'] = stock_data['60MvAvg'] - (stock_data['60StdDev'] * 2)
    
#     stock_data['20VolAvg'] = stock_data['Volume'].rolling(window=20).mean()
#     stock_data['60VolAvg'] = stock_data['Volume'].rolling(window=60).mean()

#     for index, row in stock_data.iterrows():
#         if np.isnan(row["Open"]) == True:
#             continue
#         if np.isnan(row["High"]) == True:
#             continue
#         if np.isnan(row["Low"]) == True:
#             continue
#         if np.isnan(row["Close"]) == True:
#             continue
#         if np.isnan(row["Change"]) == True:
#             continue
#         if np.isnan(row["Volume"]) == True:
#             continue
#         if np.isnan(row["5MvAvg"]) == True:
#             continue
#         if np.isnan(row["20MvAvg"]) == True:
#             continue
#         if np.isnan(row["60MvAvg"]) == True:
#             continue
#         if np.isnan(row["112MvAvg"]) == True:
#             continue
#         if np.isnan(row["224MvAvg"]) == True:
#             continue
#         if np.isnan(row["UpperBand"]) == True:
#             continue
#         if np.isnan(row["LowerBand"]) == True:
#             continue
#         if np.isnan(row["20VolAvg"]) == True:
#             continue
#         if np.isnan(row["60VolAvg"]) == True:
#             continue
#         data.append([row["Open"], row["High"], row["Low"], row["Close"], row["Change"], row["Volume"], row["5MvAvg"], row["20MvAvg"], row["60MvAvg"], row["112MvAvg"], row["224MvAvg"], row["UpperBand"], row["LowerBand"], row["20VolAvg"], row["60VolAvg"] ])
#     if len(data) > 244:
#         save_list_to_csv("%s\\dataset\\%s.csv" % (dir, code), data)
#         #write_log(f"{name} ({code}) was created!")
#         write_log(f"({code}) was created!")
#     else:
#         #write_log(f"{name} ({code}) was not created!")
#         write_log(f"({code}) was not created!")
        
# def create_sequences(data_input, data_target, seq_length, predict_days):
#     x = []
#     y = []
#     for i in range(len(data_input) - seq_length - predict_days + 1):
#         x.append(data_input[i:i+seq_length])
#         y.append(data_target[i+seq_length:i+seq_length+predict_days])
#     return np.array(x), np.array(y)

def create_machine():
    # https://roytravel.tistory.com/103
    # LSTM 모델 생성
    model_target = Sequential()
    # # 첫 번째 LSTM 레이어
    # model_target.add(LSTM((len(input_column)+sequence_length) + predict_days , return_sequences=True, input_shape=(sequence_length, len(input_column))))
    # # 두 번째 LSTM 레이어
    # model_target.add(LSTM((len(input_column)+sequence_length), return_sequences=False))
    # # 추가적인 Dense 레이어 (옵션)
    # for i in range(len(input_column)):
    #     #model_target.add(Dense((sequence_length-i) * (len(input_column)-i) * (sequence_length-i) * (len(input_column)-i))) # 더 많은 노드를 갖는 추가적인 Dense 레이어도 추가할 수 있습니다.
    #     #structure = int((sequence_length + len(input_column) - (i * 10 * weight)) * weight)
    #     #structure = int(sequence_length * len(input_column) * predict_days / (weight * (i + 1)))
    #     structure = int(pow(predict_days * (len(input_column) - i),weight))
    #     model_target.add(Dense(structure)) # 더 많은 노드를 갖는 추가적인 Dense 레이어도 추가할 수 있습니다.
    # # 출력 레이어
    # model_target.add(Dense(predict_days))

    # 미세한 오차가 발생하네... 여기 조정하면 해결 될꺼 같은데..
    # Dropout 레이어를 조정하면 해결 될꺼 같은데..
    
    # 첫 번째 LSTM 레이어
    model_target.add(LSTM((len(input_column)+sequence_length) + (predict_days * weight2) , return_sequences=True, input_shape=(sequence_length, len(input_column))))  # units는 LSTM 레이어의 뉴런 수입니다.

    # 첫 번째 Dropout 레이어
    model_target.add(Dropout(0.1))  # dropout_rate는 Dropout 비율입니다.

    # 두 번째 LSTM 레이어
    model_target.add(LSTM((len(input_column)+sequence_length), return_sequences=False))

    # 두 번째 Dropout 레이어
    model_target.add(Dropout(rate=0.1))

    # 추가적인 Dense 레이어 (옵션)
    for i in range(int(len(input_column))):
        if i % 2 == 0:
            continue
        #model_target.add(Dense((sequence_length-i) * (len(input_column)-i) * (sequence_length-i) * (len(input_column)-i))) # 더 많은 노드를 갖는 추가적인 Dense 레이어도 추가할 수 있습니다.
        #structure = int((sequence_length + len(input_column) - (i * 10 * weight)) * weight)
        #structure = int(sequence_length * len(input_column) * predict_days / (weight * (i + 1)))
        structure = int(pow((predict_days * weight2) * (len(input_column) - i),weight))
        model_target.add(Dense(structure)) # 더 많은 노드를 갖는 추가적인 Dense 레이어도 추가할 수 있습니다.
        if i % 3 == 0:
            model_target.add(Dropout(rate=0.1))
    model_target.add(Dropout(rate=0.1))
    # 출력 레이어
    model_target.add(Dense(predict_days))

    # 모델 컴파일
    # model_target.compile(optimizer='adam', loss='mean_squared_error')
    model_target.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss=tf.compat.v1.losses.mean_squared_error, metrics=["accuracy"])
    model_target.save(f'{dir}\\machine\\{machine_name}.h5')
    write_log("{machine_name}.keras was created.")
    model_target.summary()
    return model_target
    
# def learn_machine(model_target):
#     # 새로운 데이터 불러오기
#     codelist = pd.read_csv(f'{dir}\\dataset\\list.csv')
#     index = 0
#     for code in codelist["Code"]:
#         try:
#             write_log("index - " + str(index))
#             write_log(code + " is load!!")
#             index = index+1
            
#             # 새로운 데이터 불러오기
#             new_data = pd.read_csv(f'{dir}\\dataset\\{code}.csv')
#             # 필요한 열 선택 (Open, High, Low, Close, Change, Volume, 5MvAvg, 20MvAvg, 60MvAvg, 112MvAvg, 224MvAvg, UpperBand, LowerBand, 20VolAvg, 60VolAvg)
#             new_data_input = new_data[input_column]
#             # target을 두개로 잡으면 아래 Dense에서 에러가 발생한다. predict_days가 두개라서??
#             # new_data_target = new_data[['Close', 'Volume']]
#             new_data_target = new_data[target_column]

#             # 데이터 정규화
#             scaler_input = MinMaxScaler()
#             scaler_target = MinMaxScaler()

#             # 입력과 타겟 데이터 정규화
#             data_input_normalized = scaler_input.fit_transform(new_data_input)
#             data_target_normalized = scaler_target.fit_transform(new_data_target)

#             x_input, y_target = create_sequences(data_input_normalized, data_target_normalized, sequence_length, predict_days)
            
#             # 모델 학습 (1번 선행학습)
#             model_target.fit(x_input, y_target, batch_size=64, epochs=1)
            
#             # 모델 저장
#             model_target.save(f'{dir}\\machine\\{machine_name}.h5')
#             current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#             model_target.save(f'{dir}\\machine\\backup\\{machine_name}_{current_time}.h5')
#         except FileNotFoundError:
#             # 파일을 찾을 수 없을 때 실행할 코드
#             print("not file " + code)
    
if __name__ == "__main__":
    
    check_directory(dir)
    check_directory(f"{dir}\\log")
    check_directory(f"{dir}\\dataset")
    check_directory(f"{dir}\\machine")
    check_directory(f"{dir}\\machine\\backup")
    
    # logger 불러오기
    logger = setup_custom_logger('create_stock_ai')
    
    # stocks = [["Code", "Name", "Market"]]
    # for index, row in fdr.StockListing('KRX').iterrows():
    #     if row["Market"] == 'KOSPI' or row["Market"] == 'KOSDAQ':
    #         stocks.append([row["Code"], row["Name"], row["Market"]])
    # save_list_to_csv("%s\\dataset\\list.csv" % dir, stocks)
    # write_log("list was created")

    # # 스레드 풀 생성
    # #with ThreadPoolExecutor(max_workers=5) as executor:
    #     # 각 주식에 대해 병렬로 작업 실행
    #     #executor.map(process_stock_data, stocks[1:])
    # for stock in stocks[1:]:
    #     process_stock_data(stock)
        
    model_target = create_machine()
    # learn_machine(model_target)   
    
