import os
import pandas as pd
import glob
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from pypfopt import HRPOpt, CovarianceShrinkage

def process_data(files, start_date, end_date):
    df_list = []
    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, index_col='Date', parse_dates=True)[['Close']]
        df.columns = [stock_name]
        df = df[~df.index.duplicated(keep='first')].sort_index()  # 确保索引唯一且排序
        df = df.loc[start_date:end_date]  # 筛选指定时间范围
        if not df.empty:
            df_list.append(df)

    if not df_list:
        print("No data available after filtering. Please check your date range and data files.")
        return pd.DataFrame()

    df_prices = pd.concat(df_list, axis=1).ffill().bfill()  # 合并DataFrame并填充缺失值
    returns = df_prices.pct_change().dropna()  # 计算日收益率并移除NaN值
    return returns

# 设置文件夹路径和日期范围
folder_path = 'C:/Users/User/Desktop/台股資料'
start_date = '2012-01-01'
end_date = '2017-12-31'

# 读取CSV文件并处理数据
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
returns = process_data(csv_files, start_date, end_date)

if returns.empty:
    raise ValueError("Processed data is empty. Cannot proceed with correlation and dendrogram.")

# 计算相关性矩阵并转换为距离矩阵
corr_matrix = returns.corr()
dist_matrix = np.sqrt((1 - corr_matrix) / 2.0)

# 构建关联树
linkage_matrix = linkage(dist_matrix, 'ward')

# 绘制关联树
plt.figure(figsize=(15, 10))
dendrogram(linkage_matrix, labels=returns.columns)
plt.xticks(rotation=90)
plt.show()
