import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

'''
# 시계열 데이터를 t 시점과 t+1 시점으로 분할
X, y = [], []
sequence_length = 448 # 시퀀스 길이

output_dir = "D:\\work\\dataset"
dir = f"{output_dir}\\list.csv"
data = pd.read_csv(dir, encoding='utf-8')
code_list = data['Code'].tolist()

for code in code_list[1:2]:
    selected_file_name = f'{output_dir}\\{code}.csv'
    data = pd.read_csv(selected_file_name)
    
    # 데이터 전처리: 스케일링    
    scaler = MinMaxScaler()
    data[['Open','High','Low','Close','Volume']] = scaler.fit_transform(data[['Open','High','Low','Close','Volume']])
    
    for i in range(len(data) - sequence_length - 20):
        X.append(data.iloc[i:(i + sequence_length)].values)
        y.append(data.iloc[i + sequence_length + 20])  # 20일 데이터 예측

# 데이터를 학습용과 테스트용으로 분할
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)

# TensorFlow 모델 생성 (간단한 예시)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1)
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)
'''

# 데이터 불러오기
data = pd.read_csv('D:\\work\\dataset\\005930.csv')

# 필요한 열 선택 (Open, High, Low, Close, Volume)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# 데이터 정규화
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# 입력 데이터와 타겟 데이터 생성
def create_sequences(data, seq_length, predict_days):
    x = []
    y = []
    for i in range(len(data) - seq_length - predict_days + 1):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+predict_days])
    return np.array(x), np.array(y)

sequence_length = 448  # x축에 들어갈 데이터 길이
predict_days = 20  # 예측할 날짜

x, y = create_sequences(data_normalized, sequence_length, predict_days)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)

model = Sequential([LSTM(50, return_sequences=True, input_shape=(sequence_length, 5)),
                    LSTM(50, return_sequences=False),
                    Dense(predict_days * 5)])

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(x_train, y_train, batch_size=64, epochs=100)

# 테스트 데이터를 사용하여 모델 평가
predictions = model.predict(x_test)
predictions = predictions.reshape(-1, predict_days, 5)  # 예측 결과를 20일 * 5개 열로 변형
predictions = scaler.inverse_transform(predictions)  # 정규화 된 값을 다시 원래 스케일로 되돌림

print(predictions)