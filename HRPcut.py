import os
import pandas as pd
import numpy as np
import glob
from pypfopt import HRPOpt
from pypfopt.expected_returns import mean_historical_return

def process_data(files, start_date, end_date):
    df_list = []
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
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

        # 确保数据包含分析的起始日期
        if df.index.min() > start_date:
            print(f"Data for {stock_name} starts after the analysis start date, excluding.")
            continue

        # 筛选指定日期范围的数据
        df = df.loc[start_date:end_date]

        if not df.empty:
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

    if df_prices.empty:
        print("DataFrame is empty after processing. No data available for the given date range.")
        return pd.DataFrame()

    df_prices.dropna(inplace=True)
    returns = df_prices.pct_change().dropna()

    return returns


#folder_path = 'C:/Users/User/Desktop/台股資料/負相關全部/負的多的/大於0.01'
folder_path = 'C:/Users/User/Desktop/台股資料/負相關全部/負的多的/大於0.01'
csv_files = glob.glob(folder_path + "/*.csv")

returns_2012_2017 = process_data(csv_files, "2012-05-02", "2015-12-31")
returns_2018_2023 = process_data(csv_files, "2018-01-01", "2023-11-15")

cov_matrix_pd_2012_2017 = returns_2012_2017.cov()
cov_matrix_np_2012_2017 = np.cov(returns_2012_2017.values.T)

cov_matrix_pd_2018_2023 = returns_2018_2023.cov()
cov_matrix_np_2018_2023 = np.cov(returns_2018_2023.values.T)

hrp_2012_2017 = HRPOpt(returns_2012_2017,cov_matrix_np_2012_2017)
hrp_2018_2023 = HRPOpt(returns_2018_2023,cov_matrix_np_2018_2023)

hrp_2012_2017.optimize()
hrp_2018_2023.optimize()

cleaned_weights_2012_2017 = hrp_2012_2017.clean_weights()
cleaned_weights_2018_2023 = hrp_2018_2023.clean_weights()

# 输出优化权重到CSV
weights_2012_2017_df = pd.DataFrame(list(cleaned_weights_2012_2017.items()), columns=['Ticker', 'Weight'])
weights_2018_2023_df = pd.DataFrame(list(cleaned_weights_2018_2023.items()), columns=['Ticker', 'Weight'])

weights_2012_2017_df.to_csv('C:/Users/User/Desktop/123/權重/Optimal_Weights_2012_2015負0.01.csv', index=False)
#weights_2018_2023_df.to_csv('C:/Users/User/Desktop/123/權重/Optimal_Weights_2018_20230.2.csv', index=False)


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
