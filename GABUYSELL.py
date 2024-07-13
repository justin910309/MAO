import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime

def read_weights_from_csv(csv_file_path):
    """从CSV文件中读取权重数据，并将其转换为pandas Series以方便操作"""
    weights_df = pd.read_csv(csv_file_path, index_col='Ticker')
    weights = weights_df['Weight']
    return weights

def process_data(files, start_date, end_date):
    df_list = []
    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        df = df[['Close']]
        df.columns = [stock_name]
        #print(f"Processing {stock_name} with data from {df.index.min()} to {df.index.max()}")
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
        df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max())).ffill()
        df = df.loc[start_date:end_date]
        if not df.empty:
            df_list.append(df)
    if not df_list:
        #print("No data available after processing. Please check your files and date range.")
        return pd.DataFrame()
    df_prices = pd.concat(df_list, axis=1).ffill().bfill()
    return df_prices

def calculate_investment_performance(weights, initial_investment, price_data):
    # Convert weights dictionary to Series and align with price_data columns
    weights = pd.Series(weights, index=price_data.columns).fillna(0)
    initial_shares = initial_investment * weights / price_data.iloc[0]
    final_values = initial_shares * price_data.iloc[-1]
    total_final_value = final_values.sum()
    total_return = total_final_value - initial_investment
    return_percentage = (total_return / initial_investment) * 100
    
    # 计算年化回报
    annualized_return = ((1 + total_return/initial_investment) ** (252/len(price_data))) - 1
    
    # 计算夏普比率，假设无风险利率为0.01（1%）
    risk_free_rate = 0.01
    daily_returns = price_data.pct_change().dot(weights)
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    
    # 计算最大回撤
    cumulative_returns = (1 + daily_returns).cumprod()
    drawdown = (cumulative_returns / cumulative_returns.cummax()) - 1
    max_drawdown = drawdown.min()
    
    return {
        'Total Final Value': total_final_value,
        'Total Return': total_return,
        'Return Percentage': return_percentage,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

# 例子
weights_csv_path = 'C:/Users/User/Desktop/123/權重/GA/Optimal_Weights_2012_2015GA台p300.csv'
data_folder_path = 'C:/Users/User/Desktop/台股資料/'
csv_files = glob.glob(os.path.join(data_folder_path, '*.csv'))
df_prices = process_data(csv_files, "2018-01-01", "2022-12-31")
weights = read_weights_from_csv(weights_csv_path)
performance_metrics = calculate_investment_performance(weights, 100000, df_prices)
print(performance_metrics)
