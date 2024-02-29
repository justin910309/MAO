import os
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import CovarianceShrinkage
# Function to read stock data from folder
def read_stock_data_from_folder(folder_path):
    stock_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            stock_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(folder_path, file_name)
            stock_data[stock_name] = pd.read_csv(file_path)
    return stock_data

# Folder path containing CSV files
folder_path = r'C:\Users\User\Desktop\台股資料\test_files'

# Read stock data from folder
stock_data = read_stock_data_from_folder(folder_path)

# Concatenate 'Close' column of all stocks into one DataFrame
df = pd.concat([stock_data[stock_name]['Close'].replace('-', np.nan).rename(stock_name).astype(float) for stock_name in stock_data], axis=1)

# Drop rows with NaN values
df = df.dropna()

# Calculate expected returns and covariance matrix
mu = expected_returns.mean_historical_return(df)
S=CovarianceShrinkage(df).ledoit_wolf() #投資報酬率高得離譜
#S = risk_models.sample_cov(df)

# Set risk-free rate
risk_free_rate = min(mu) + 0.001

# Check if covariance matrix is positive semi-definite
if not np.all(np.linalg.eigvals(S) > 0):
    print("Warning: Covariance matrix is not positive semi-definite. Please check your data.")

# Construct efficient frontier with solver specified
ef = EfficientFrontier(mu, S, solver='SCS')

ef.add_constraint(lambda w: w.sum() == 1)  # 總權重為1

# Attempt to maximize Sharpe ratio
try:
    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()
    ef.portfolio_performance(verbose=True)
    print("\nOptimal Weights:")
    print(cleaned_weights)
except Exception as e:
    print("An error occurred while optimizing portfolio:", e)
