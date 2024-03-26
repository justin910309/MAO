import os
import pandas as pd
import numpy as np
import glob
from pypfopt import HRPOpt
from pypfopt.expected_returns import mean_historical_return

def process_data(files, start_date, end_date):
    df_list = []
    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        df = df[['Close']]
        df.columns = [stock_name]

        # 确保索引是单调递增的
        df = df.sort_index()

        # 检查并处理重复的索引（日期）
        if df.index.duplicated().any():
            print(f"Found duplicate dates in {stock_name}, taking the first occurrence.")
            df = df[~df.index.duplicated(keep='first')]

        # 筛选指定日期范围的数据
        if not df.loc[start_date:end_date].empty:
            df = df.loc[start_date:end_date]
            df_list.append(df)
        else:
            print(f"No data available for {stock_name} in the given date range.")

    if not df_list:
        print("No data available after filtering. Please check your date range and data files.")
        return pd.DataFrame()

    df_prices = pd.concat(df_list, axis=1)
    df_prices.ffill(inplace=True)
    df_prices.bfill(inplace=True)
    df_prices.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 在丢弃NaN之前检查df_prices是否为空
    if df_prices.empty:
        print("DataFrame is empty after processing. No data available for the given date range.")
        return pd.DataFrame()

    df_prices.dropna(inplace=True)
    returns = df_prices.pct_change().dropna()

    return returns

folder_path = 'C:/Users/User/Desktop/台股資料'
csv_files = glob.glob(folder_path + "/*.csv")

returns_2012_2017 = process_data(csv_files, "2012-05-02", "2017-12-31")
returns_2018_2023 = process_data(csv_files, "2018-01-01", "2023-11-15")

# 使用HRPOpt进行优化
hrp_2012_2017 = HRPOpt(returns_2012_2017)
hrp_2018_2023 = HRPOpt(returns_2018_2023)

hrp_2012_2017.optimize()
hrp_2018_2023.optimize()

cleaned_weights_2012_2017 = hrp_2012_2017.clean_weights()
cleaned_weights_2018_2023 = hrp_2018_2023.clean_weights()

# 输出优化权重到CSV
weights_2012_2017_df = pd.DataFrame(list(cleaned_weights_2012_2017.items()), columns=['Ticker', 'Weight'])
weights_2018_2023_df = pd.DataFrame(list(cleaned_weights_2018_2023.items()), columns=['Ticker', 'Weight'])

weights_2012_2017_df.to_csv('Optimal_Weights_2012_2017-1.csv', index=False)
weights_2018_2023_df.to_csv('Optimal_Weights_2018_2023-1.csv', index=False)

# 显示投资组合性能，并保存性能指标到CSV
perf_2012_2017 = hrp_2012_2017.portfolio_performance(verbose=True)
perf_2018_2023 = hrp_2018_2023.portfolio_performance(verbose=True)

perf_metrics = pd.DataFrame({
    'Period': ['2012-2017', '2018-2023'],
    'Expected Annual Return': [perf_2012_2017[0], perf_2018_2023[0]],
    'Annual Volatility': [perf_2012_2017[1], perf_2018_2023[1]],
    'Sharpe Ratio': [perf_2012_2017[2], perf_2018_2023[2]]
})

perf_metrics.to_csv('Portfolio_Performance_Metrics.csv', index=False)
