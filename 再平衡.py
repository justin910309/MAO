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

def rebalance_portfolio(df_prices, weights, initial_investment=100000, rebalance_period='365D', commission_rate=0.001425, tax_rate=0.003):
    current_value = initial_investment
    rebalance_dates = pd.date_range(start=df_prices.index.min(), end=df_prices.index.max(), freq=rebalance_period)
    
    returns_list = []
    daily_values = [initial_investment]

    for i in range(len(rebalance_dates)-1):
        period_start = rebalance_dates[i]
        period_end = rebalance_dates[i+1] if i < len(rebalance_dates)-2 else df_prices.index.max()

        if period_start not in df_prices.index or period_end not in df_prices.index:
            continue

        # Calculate investments per stock at the start of the period
        investment_per_stock = {stock: current_value * weight for stock, weight in weights.items()}
        prices_at_start = df_prices.loc[period_start]
        shares = {stock: investment / prices_at_start[stock] for stock, investment in investment_per_stock.items() if prices_at_start[stock] > 0}

        # Subtract commission for buying
        buying_commission = sum(investment_per_stock[stock] * commission_rate for stock in shares)
        current_value -= buying_commission

        # Calculate values and returns within the period
        period_values = df_prices.loc[period_start:period_end].apply(lambda prices: sum(shares[stock] * prices[stock] for stock in shares), axis=1)
        daily_values.extend(period_values[1:].tolist())
        daily_returns = pd.concat([pd.Series(daily_values).pct_change().dropna()])

        # Calculate values at the end of the period
        prices_at_end = df_prices.loc[period_end]
        value_at_end = sum(shares[stock] * prices_at_end.get(stock, 0) for stock in shares)

        # Subtract commission and taxes for selling
        selling_commission = sum(shares[stock] * prices_at_end.get(stock, 0) * commission_rate for stock in shares if stock in prices_at_end)
        taxes = sum(shares[stock] * prices_at_end.get(stock, 0) * tax_rate for stock in shares if stock in prices_at_end)
        value_at_end -= (selling_commission + taxes)

        current_value = value_at_end

    # Calculate total return
    total_return = current_value - initial_investment

    # Calculate return percentage
    return_percentage = (total_return / initial_investment) * 100

    # Calculate annualized return
    annualized_return = ((current_value / initial_investment) ** (252 / len(df_prices))) - 1

    # Calculate Sharpe ratio, assuming a risk-free rate of 0.01 (1%)
    risk_free_rate = 0.01
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    # Calculate maximum drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    drawdown = (cumulative_returns / cumulative_returns.cummax()) - 1
    max_drawdown = drawdown.min()

    performance_metrics = {
        'Total Final Value': current_value,
        'Total Return': total_return,
        'Return Percentage': return_percentage,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

    print("Performance Metrics with Rebalance:", performance_metrics)
    return performance_metrics

# 更新文件路径
weights_csv_path = 'C:/Users/User/Desktop/123/權重/Optimal_Weights_2012_2015負all.csv'
data_folder_path = 'C:/Users/User/Desktop/台股資料/'

# 读取权重和处理数据
weights = read_weights_from_csv(weights_csv_path)
csv_files = glob.glob(os.path.join(data_folder_path, '*.csv'))

# 处理数据并计算收益
df_prices = process_data(csv_files, "2012-05-02", "2015-12-31")

if not df_prices.empty:
    performance_metrics = rebalance_portfolio(df_prices, weights, initial_investment=100000, rebalance_period='365D')
else:
    print("No data available for the specified period.")
