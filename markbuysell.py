import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta

def read_weights_from_csv(csv_file_path):
    """从CSV文件中读取权重数据"""
    weights_df = pd.read_csv(csv_file_path)
    weights = dict(zip(weights_df['Ticker'], weights_df['Weight']))
    return weights

def process_data(files, start_date, end_date):
    df_list = []
    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        df = df[['Close']]
        df.columns = [stock_name]

        # 确保索引没有重复的日期
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
        
        # 创建完整日期范围的DatetimeIndex
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max())

        # 使用reindex填充缺失的日期，并前向填充缺失值
        df = df.reindex(full_date_range).ffill()

        # 筛选指定日期范围的数据
        df = df.loc[start_date:end_date]

        if not df.empty:
            df_list.append(df)

    if df_list:
        df_prices = pd.concat(df_list, axis=1)
        df_prices.ffill(inplace=True)
        df_prices.bfill(inplace=True)
        return df_prices
    else:
        print("No data available after processing. Please check your files and date range.")
        return pd.DataFrame()

def calculate_portfolio_performance(df_prices, weights, initial_investment=100000, commission_rate=0.001425, tax_rate=0.003):
    # 确保权重中的股票与价格数据中的股票一致
    df_prices = df_prices[list(weights.keys())]

    # Calculate the initial shares for each stock
    investment_per_stock = {stock: initial_investment * weight for stock, weight in weights.items()}
    prices_at_start = df_prices.iloc[0]
    shares = {stock: investment / prices_at_start[stock] for stock, investment in investment_per_stock.items() if prices_at_start[stock] > 0}

    # Subtract commission for buying
    buying_commission = sum(investment_per_stock[stock] * commission_rate for stock in shares)
    current_value = initial_investment - buying_commission
    
    # Calculate the portfolio value at the end
    prices_at_end = df_prices.iloc[-1]
    value_at_end = sum(shares[stock] * prices_at_end.get(stock, 0) for stock in shares)

    # Subtract commission and taxes for selling
    selling_commission = sum(shares[stock] * prices_at_end.get(stock, 0) * commission_rate for stock in shares if stock in prices_at_end)
    taxes = sum(shares[stock] * prices_at_end.get(stock, 0) * tax_rate for stock in shares if stock in prices_at_end)
    value_at_end -= (selling_commission + taxes)

    # Calculate performance metrics
    total_final_value = value_at_end
    total_return = total_final_value - initial_investment
    return_percentage = (total_return / initial_investment) * 100
    annualized_return = ((total_final_value / initial_investment) ** (252 / len(df_prices))) - 1
    daily_returns = df_prices.pct_change().dropna().dot(pd.Series(weights))
    risk_free_rate = 0.01
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    cumulative_returns = (1 + daily_returns).cumprod()
    drawdown = (cumulative_returns / cumulative_returns.cummax()) - 1
    max_drawdown = drawdown.min()

    performance_metrics = {
        'Total Final Value': total_final_value,
        'Total Return': total_return,
        'Return Percentage': return_percentage,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

    print("Performance Metrics:", performance_metrics)
    return performance_metrics

# 更新这些路径以匹配您的文件位置
weights_csv_path = 'C:/Users/User/Desktop/123/權重/Optimal_Weights_2012_2015all.csv'
data_folder_path = 'C:/Users/User/Desktop/台股資料/'

# 读取权重和处理数据
weights = read_weights_from_csv(weights_csv_path)
csv_files = glob.glob(os.path.join(data_folder_path, '*.csv'))

# 处理2016-2017年的数据并计算回报
df_prices = process_data(csv_files, "2012-05-02", "2015-12-31")

if not df_prices.empty:
    performance_metrics = calculate_portfolio_performance(df_prices, weights, initial_investment=100000)
else:
    print("No data available for the specified period.")
