
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 获取股市数据
def fetch_data(stock_symbol, start_date='2020-01-01', end_date='2023-01-01'):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# 特征工程：计算技术指标（例如移动平均线、相对强弱指数）
def add_technical_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # 50日简单移动平均线
    df['SMA_200'] = df['Close'].rolling(window=200).mean()  # 200日简单移动平均线
    df['RSI'] = compute_rsi(df['Close'], 14)  # 计算相对强弱指数
    return df

# 计算RSI（相对强弱指数）
def compute_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 创建目标变量：上涨或下跌（以收盘价变化来判断）
def create_target_variable(df):
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # 1表示上涨，0表示下跌
    return df

# 训练机器学习模型
def train_model(df):
    df = df.dropna()  # 删除缺失值
    X = df[['SMA_50', 'SMA_200', 'RSI']]  # 特征
    y = df['Target']  # 目标变量
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    return model

# 执行投资决策
def make_investment_decision(df, model):
    latest_data = df.iloc[-1][['SMA_50', 'SMA_200', 'RSI']].values.reshape(1, -1)
    prediction = model.predict(latest_data)
    
    if prediction == 1:
        decision = "Buy"
    else:
        decision = "Sell"
    
    return decision

# 主程序
def main():
    stock_symbol = 'AAPL'  # 设置您感兴趣的股票代码
    data = fetch_data(stock_symbol)
    data = add_technical_indicators(data)
    data = create_target_variable(data)
    
    # 训练模型
    model = train_model(data)
    
    # 做出投资决策
    decision = make_investment_decision(data, model)
    print(f"Investment Decision for {stock_symbol}: {decision}")
    
    # 可视化股票价格与技术指标
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['SMA_50'], label='50-Day SMA')
    plt.plot(data['SMA_200'], label='200-Day SMA')
    plt.title(f'{stock_symbol} Stock Price and Technical Indicators')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
