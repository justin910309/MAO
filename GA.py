import pandas as pd
import numpy as np
import os
import glob
import random

def process_data(files, start_date, end_date):
    df_list = []
    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        df = df[['Close']]
        df.columns = [stock_name]
        df = df.sort_index()
        if df.index.duplicated().any():
            print(f"Found duplicate dates in {stock_name}, taking the first occurrence.")
            df = df[~df.index.duplicated(keep='first')]
        if not df.loc[start_date:end_date].empty:
            df = df.loc[start_date:end_date]
            df_list.append(df)
        else:
            print(f"No data available for {stock_name} in the given date range.")
    if not df_list:
        print("No data available after filtering. Please check your date range and data files.")
        return pd.DataFrame(), pd.DataFrame()
    df_prices = pd.concat(df_list, axis=1)
    df_prices.ffill(inplace=True)
    df_prices.bfill(inplace=True)
    df_prices.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df_prices.empty:
        print("DataFrame is empty after processing. No data available for the given date range.")
        return pd.DataFrame(), pd.DataFrame()
    df_prices.dropna(inplace=True)
    returns = df_prices.pct_change().dropna()
    return returns, df_prices

def initialize_population(size, num_stocks):
    return np.array([np.random.dirichlet(np.ones(num_stocks)) for _ in range(size)])

def fitness_function(chromosome, returns):
    portfolio_return = np.dot(returns, chromosome)
    return portfolio_return.mean() / portfolio_return.std()

def select(population, fitness_scores):
    fitness_scores = fitness_scores / np.sum(fitness_scores)
    return np.array(random.choices(population, weights=fitness_scores, k=len(population)))

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    child1 /= np.sum(child1)  # Normalize
    child2 /= np.sum(child2)  # Normalize
    return child1, child2

def mutate(chromosome, rate=0.1):
    if random.random() < rate:
        mutation_idx = random.randint(0, len(chromosome) - 1)
        mutation_value = np.random.normal(loc=chromosome[mutation_idx], scale=0.1)
        chromosome[mutation_idx] += mutation_value  # Use += to adjust rather than replace
        chromosome = np.maximum(chromosome, 0)  # Ensure no negative weights
        chromosome /= np.sum(chromosome)  # Normalize to make sum equal to 1
    return chromosome

def genetic_algorithm(returns, iterations=50, population_size=100):
    num_stocks = returns.shape[1]
    population = initialize_population(population_size, num_stocks)
    for i in range(iterations):
        fitness_scores = np.array([fitness_function(chromo, returns) for chromo in population])
        population = select(population, fitness_scores)
        next_generation = []
        for j in range(0, len(population), 2):
            child1, child2 = crossover(population[j], population[j+1])
            next_generation.append(mutate(child1))
            next_generation.append(mutate(child2))
        population = np.array(next_generation)
    best_idx = np.argmax([fitness_function(chromo, returns) for chromo in population])
    return population[best_idx]

def calculate_portfolio_value(best_portfolio, initial_investment, price_data):
    # Initialize investments based on portfolio weights
    initial_amounts = initial_investment * best_portfolio
    initial_shares = initial_amounts / price_data.iloc[0]
    # Calculate the final value of the investments
    final_values = initial_shares * price_data.iloc[-1]
    total_final_value = final_values.sum()
    return total_final_value

# Setup folder path and run
folder_path = 'C:/Users/user/Desktop/台股資料/負相關全部'
csv_files = glob.glob(folder_path + "/*.csv")
start_date = pd.to_datetime('2018-01-01')
end_date = pd.to_datetime('2022-12-31')
stock_returns, price_data = process_data(csv_files, start_date, end_date)

if not stock_returns.empty:
    best_portfolio = genetic_algorithm(stock_returns)
    initial_investment = 100000
    final_value = calculate_portfolio_value(best_portfolio, initial_investment, price_data)
    print("Best Portfolio:", best_portfolio)
    print(f"Initial Investment: {initial_investment}")
    print(f"Final Value of Portfolio: {final_value}")
    print(f"Return on Investment: {final_value - initial_investment}")
else:
    print("No data to process.")

if not csv_files:
    print("No CSV files found in the directory.")
else:
    print(f"Found {len(csv_files)} files.")
