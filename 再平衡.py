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




def rebalance_portfolio(df_prices, weights, initial_investment=100000, rebalance_period='120D'):
    current_value = initial_investment
    rebalance_dates = pd.date_range(start=df_prices.index.min(), end=df_prices.index.max(), freq=rebalance_period)
    
    returns_list = []

    for i in range(len(rebalance_dates)-1):
        period_start = rebalance_dates[i]
        period_end = rebalance_dates[i+1] if i < len(rebalance_dates)-2 else df_prices.index.max()

        if period_start not in df_prices.index:
            continue

        investment_per_stock = {stock: current_value * weight for stock, weight in weights.items()}
        prices_at_start = df_prices.loc[period_start]
        shares = {stock: investment / prices_at_start[stock] for stock, investment in investment_per_stock.items()}

        prices_at_end = df_prices.loc[period_end]
        value_at_end = sum(shares[stock] * prices_at_end.get(stock, 0) for stock in shares)

        period_return = (value_at_end - current_value) / current_value
        returns_list.append(period_return)

        current_value = value_at_end

    total_return = sum(returns_list)
    print("Periodic returns:", returns_list)
    print("Total return:", total_return)
    # 打印最终投资组合的价值
    print("Final portfolio value:", current_value)

    return returns_list, total_return, current_value


# Update these paths to match your file locations
weights_csv_path = 'C:/Users/User/Desktop/123/Optimal_Weights_2012_2017-1.csv'
data_folder_path = 'C:/Users/User/Desktop/台股資料'

# Read weights and process data
weights = read_weights_from_csv(weights_csv_path)
csv_files = glob.glob(os.path.join(data_folder_path, '*.csv'))

# Process data for 2018 and calculate returns
df_prices_2018 = process_data(csv_files, "2018-01-01", "2020-12-31")
if not df_prices_2018.empty:
    returns_list, total_return, final_portfolio_value = rebalance_portfolio(df_prices_2018, weights, initial_investment=100000, rebalance_period='120D')
    print("Periodic returns:", returns_list)
    print("Total return:", total_return)
    print("Final portfolio value:", final_portfolio_value)
else:
    print("No data available for the specified period.")


# 计算日收益率
returns_2018 = df_prices_2018.pct_change().dropna()

# 计算相关性矩阵
correlation_matrix = returns_2018.corr()
output_file_path = 'C:/Users/User/Desktop/123/correlation_matrix.csv'

# 将相关性矩阵导出到CSV文件
#correlation_matrix.to_csv(output_file_path)
# 打印相关性矩阵
print("Correlation Matrix:")
print(correlation_matrix)


