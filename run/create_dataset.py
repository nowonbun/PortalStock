import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import os
import numpy as np
import logging

'''
시작 전에 dataset 폴더와 그 안의 backup 폴더를 생성해야 한다.
추후 기능 추가
'''
output_dir = "c:\\stock"
start_date = "2013-01-01"
end_date = "2024-01-07"
logger = None
#end_date = "2024-01-07"

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

    handler = logging.FileHandler(f'{output_dir}\\log\\logfile_{name}.log')  # 로그 파일 이름 및 경로 지정
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger

def write_log(msg):
    logger.info(msg)
    print(msg)
        
def process_stock_data(stock):
    code, name, market = stock
    #data = [["Code", "Date", "Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "448MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg", "ConversionLine", "BaseLine", "LeadingSpanA", "LaggingSpanB"]]
    #data = [["Code", "Date", "Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "448MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg"]]
    #data = [["Open", "High", "Low", "Close", "Volume"]]
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
    '''
        # 시간이 너무 오래 걸리더라..
        
        # 전환선 계산
        nine_period_high = stock_data['High'].rolling(window=9).max()
        nine_period_low = stock_data['Low'].rolling(window=9).min()
        stock_data['ConversionLine'] = (nine_period_high + nine_period_low) / 2

        # 기준선 계산
        twenty_six_period_high = stock_data['High'].rolling(window=26).max()
        twenty_six_period_low = stock_data['Low'].rolling(window=26).min()
        stock_data['BaseLine'] = (twenty_six_period_high + twenty_six_period_low) / 2

        # 선행 스패닝 계산 (전환선 + 기준선) / 2, 26일 전의 값
        stock_data['LeadingSpanA'] = ((stock_data['Conversion Line'] + stock_data['Base Line']) / 2).shift(26)

        # 후행 스패닝 계산 (현재 가격을 26일 후로 이동한 값)
        stock_data['LaggingSpanB'] = stock_data['Close'].shift(-26)
    ''' 
    for index, row in stock_data.iterrows():
        #data.append([market+code,index.strftime('%Y-%m-%d'), row["Open"], row["High"], row["Low"], row["Close"], row["Change"], row["Volume"], row["5MvAvg"], row["20MvAvg"], row["60MvAvg"], row["112MvAvg"], row["224MvAvg"], row["448MvAvg"], row["UpperBand"], row["LowerBand"], row["20VolAvg"], row["60VolAvg"], row["ConversionLine"], row["BaseLine"], row["LeadingSpanA"], row["LaggingSpanB"] ])
        #data.append([market+code,index.strftime('%Y-%m-%d'), row["Open"], row["High"], row["Low"], row["Close"], row["Change"], row["Volume"], row["5MvAvg"], row["20MvAvg"], row["60MvAvg"], row["112MvAvg"], row["224MvAvg"], row["448MvAvg"], row["UpperBand"], row["LowerBand"], row["20VolAvg"], row["60VolAvg"] ])
        #data.append([row["Open"], row["High"], row["Low"], row["Close"], row["Volume"] ])
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
        data.append([row["Open"], row["High"], row["Low"], row["Close"], row["Change"], row["Volume"], row["5MvAvg"], row["20MvAvg"], row["60MvAvg"], row["112MvAvg"], row["224MvAvg"], row["UpperBand"], row["LowerBand"], row["20VolAvg"], row["60VolAvg"], row["Close"] * row["Volume"] ])
    if len(data) > 244:
        save_list_to_csv("%s\\dataset\\%s.csv" % (output_dir, code), data)
        write_log(f"({code}) was created!")

        # 추가분 학습용
        '''
        adddata = ["Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg"] + data[-244:]
        save_list_to_csv("%s\\dataset\\%s.csv" % (output_dir, code), adddata)
        # 245행이 있는 것을 확인해야 한다.
        '''
    else:
        write_log(f"({code}) was not created!")
    
if __name__ == "__main__":
    
    check_directory(output_dir)
    check_directory(f"{output_dir}\\log")
    check_directory(f"{output_dir}\\dataset")
    
    # logger 불러오기
    logger = setup_custom_logger('create_stock_dataset')
    
    stocks = [["Code", "Name", "Market"]]
    for index, row in fdr.StockListing('KRX').iterrows():
        if row["Market"] == 'KOSPI' or row["Market"] == 'KOSDAQ':
            stocks.append([row["Code"], row["Name"], row["Market"]])
    save_list_to_csv("%s\\dataset\\list.csv" % output_dir, stocks)
    write_log("list was created")

    # 스레드 풀 생성
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 각 주식에 대해 병렬로 작업 실행
        executor.map(process_stock_data, stocks[1:])   
    
