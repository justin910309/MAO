import os
import pandas as pd
import numpy as np
import glob
from scipy.optimize import minimize

def process_data(files, start_date, end_date):
    df_list = []
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        df = df[['Close']]
        df.columns = [stock_name]
        
        df = df.sort_index()
        
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
        
        if df.index.min() > start_date:
            continue
        
        df = df.loc[start_date:end_date]

        if not df.empty:
            df_list.append(df)
        else:
            print(f"No data available for {stock_name} in the given date range.")

    if not df_list:
        print("No data available after filtering. Please check your date range and data files.")
        return pd.DataFrame()

    df_prices = pd.concat(df_list, axis=1)
    print(f"Combined DataFrame shape: {df_prices.shape}")

    df_prices.ffill(inplace=True)
    df_prices.bfill(inplace=True)
    df_prices.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 用均值填充NaN值
    df_prices = df_prices.apply(lambda x: x.fillna(x.mean()), axis=0)

    if df_prices.empty:
        print("DataFrame is empty after processing. No data available for the given date range.")
        return pd.DataFrame()

    returns = df_prices.pct_change().dropna()
    print(f"Returns DataFrame shape: {returns.shape}")
    return returns

def check_and_clean_data(returns, threshold=1):
    for column in returns.columns:
        col_min = returns[column].min()
        col_max = returns[column].max()
        if col_min < -threshold or col_max > threshold:
            print(f"Column {column}: min={col_min}, max={col_max}")
            returns[column] = np.clip(returns[column], -threshold, threshold)
    return returns

def calculate_mean_and_covariance(returns):
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')
    mean_returns = returns.mean()
    covariance_matrix = returns.cov()
    return mean_returns, covariance_matrix

def optimize_portfolio(mean_returns, covariance_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, covariance_matrix)
    
    def portfolio_return(weights, mean_returns):
        return np.sum(mean_returns * weights)
    
    def portfolio_volatility(weights, mean_returns, covariance_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    def negative_sharpe_ratio(weights, mean_returns, covariance_matrix, risk_free_rate=0):
        p_ret = portfolio_return(weights, mean_returns)
        p_vol = portfolio_volatility(weights, mean_returns, covariance_matrix)
        return -(p_ret - risk_free_rate) / p_vol

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

folder_path = 'C:/Users/User/Desktop/台股資料/'
csv_files = glob.glob(folder_path + "/*.csv")

returns_2012_2015 = process_data(csv_files, "2012-05-02", "2015-12-31")

# 如果這個返回的數據框是空的，則退出程序
if returns_2012_2015.empty:
    print("No valid data available for the given date range. Exiting.")
else:
    returns_2012_2015 = check_and_clean_data(returns_2012_2015, threshold=1)

    returns_2012_2015.replace([np.inf, -np.inf], np.nan, inplace=True)
    returns_2012_2015.dropna(inplace=True)

    returns_2012_2015 = returns_2012_2015.apply(pd.to_numeric, errors='coerce')

    print("NaN values in returns_2012_2015 after processing:", returns_2012_2015.isna().sum().sum())
    print("Infinite values in returns_2012_2015 after processing:", np.isinf(returns_2012_2015).sum().sum())

    print("Summary statistics for returns_2012_2015:")
    print(returns_2012_2015.describe())

    # 確保數據沒有 NaN 或無窮大值
    if returns_2012_2015.isna().sum().sum() == 0 and np.isinf(returns_2012_2015).sum().sum() == 0:
        try:
            print("Calculating mean historical returns for 2012-2015.")
            mu_2012_2015, S_2012_2015 = calculate_mean_and_covariance(returns_2012_2015)
            print("Mean historical returns for 2012-2015 calculated:")
            print(mu_2012_2015)

            print("Optimizing portfolio for 2012-2015.")
            result_2012_2015 = optimize_portfolio(mu_2012_2015, S_2012_2015)
            print("Optimized weights for 2012-2015:")
            print(result_2012_2015.x)

            weights_2012_2015_df = pd.DataFrame(list(zip(returns_2012_2015.columns, result_2012_2015.x)), columns=['Ticker', 'Weight'])
            weights_2012_2015_df.to_csv('Optimal_Weights_2012_2015.csv', index=False)
        except ValueError as e:
            print("Error during optimization or performance calculation:", e)
    else:
        print("Data still contains NaN or infinite values after processing. Please check your data.")
