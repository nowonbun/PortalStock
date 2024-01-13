import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import os
import numpy as np
import logging

output_dir = "D:\\work\\dataset"
start_date = "2020-01-01"
#end_date = "2023-10-31"
end_date = "2024-01-07"

def save_list_to_csv(file_path, data):
    if os.path.exists(file_path):
        os.remove(file_path)
        
    with open(file_path, mode='w', newline='' , encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.FileHandler(f'{output_dir}\\logfile_{name}.log')  # 로그 파일 이름 및 경로 지정
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    return logger

logger = setup_custom_logger('create_stock_dataset')
        
def process_stock_data(stock):
    code, name, market = stock
    #data = [["Code", "Date", "Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "448MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg", "ConversionLine", "BaseLine", "LeadingSpanA", "LaggingSpanB"]]
    #data = [["Code", "Date", "Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "448MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg"]]
    #data = [["Open", "High", "Low", "Close", "Volume"]]
    data = [["Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg"]]
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
        data.append([row["Open"], row["High"], row["Low"], row["Close"], row["Change"], row["Volume"], row["5MvAvg"], row["20MvAvg"], row["60MvAvg"], row["112MvAvg"], row["224MvAvg"], row["UpperBand"], row["LowerBand"], row["20VolAvg"], row["60VolAvg"] ])
    if len(data) > 224:
        target_data = [["Open", "High", "Low", "Close", "Change", "Volume", "5MvAvg", "20MvAvg", "60MvAvg", "112MvAvg", "224MvAvg", "UpperBand", "LowerBand", "20VolAvg", "60VolAvg"]] + data[-224:]
        save_list_to_csv("%s\\target_%s.csv" % (output_dir, code), target_data)
        logger.info(f"{name} ({code}) was created!")
        print(f"{name} ({code}) was created!")
    else:
        logger.info(f"{name} ({code}) was not created!")
        print(f"{name} ({code}) was not created!")
    
if __name__ == "__main__":
    # logger 불러오기
    logger = setup_custom_logger('create_stock_targetset')
    
    process_stock_data(["024900", "", ""]) 
    