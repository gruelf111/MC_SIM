import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

enddate = dt.datetime.now()
startdate = enddate - dt.timedelta(days=300)

stocklist = {'SUSW.L':0.099,
             'MSFT':0.092, 
             'KBH':0.079, 
             'NVDA':0.07, 
             'NVO':0.066, 
             'TSMC34.SA':0.039, 
             'V':0.039, 
             'TSCO':0.037, 
             'EKT.DE':0.036,
             'RR.L':0.032, 
             'AAPL':0.032, 
             'MRU.TO':0.031, 
             'APD':0.028, 
             'AIXA.DE':0.026, 
             'TSU.TO':0.023, 
             'HD':0.022, 
             'EPAM':0.018, 
             'EBAY':0.017, 
             'CWEN':0.016, 
             'MOR':0.011, 
             'SEDG':0.01, 
             'KER.PA':0.007,
             }

stocks = list(stocklist.keys())
data = yf.download(stocks,start=startdate, end=enddate)['Close'][stocks]

df = pd.DataFrame(data=data)
df = df.dropna()

log_returns = np.log(df) - np.log(df.shift(1))
weight = np.full((len(stocklist)),list(stocklist.values()))

returns = log_returns
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * np.sqrt(252)

num_portfolios = 10000

results = pd.DataFrame(columns=['returns', 'volatility', 'sharpe', 'weights'], index=range(num_portfolios))

weights_record = []

for i in range(num_portfolios):
		# Normalised randomly generated weights
    weights = np.random.random(len(stocklist))
    weights /= np.sum(weights)

		# Calculate returns and volatility
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

		# Store results
    results.loc[i, 'returns'] = returns
    results.loc[i, 'volatility'] = volatility
    results.loc[i, 'sharpe'] = results.loc[i, 'returns'] / results.loc[i, 'volatility']
    results.loc[i, 'weights'] = ','.join(str(np.round(weight, 4)) for weight in weights)
		# record weights
    weights_record.append(weights)
    
df_top10 = results.sort_values('sharpe', ascending=False).head(10)['weights']

# Initialize a dictionary to store the weights for each stock
optimal_weights = {ticker: [] for ticker in stocklist}

# Iterate over the top 10 portfolios
for weights_str in df_top10:
    weights = [float(weight) for weight in weights_str.split(',')]
    for ticker, weight in zip(stocklist, weights):
        optimal_weights[ticker].append(weight)

# Optional: Calculate the average weight for each ticker
average_weights = {ticker: round(np.mean(weights),4) for ticker, weights in optimal_weights.items()}

optimal_weight = list(np.array(list(average_weights.values()))*100)
print(optimal_weight)
tickers = list(stocklist.keys())

plt.barh(tickers, optimal_weight, color='skyblue')

for index, (ticker, value) in enumerate(average_weights.items()):
    plt.text(optimal_weight[index], index, f"{optimal_weight[index]:.1f}") 
plt.ylabel("Yahoo Finance Ticker")
plt.xlabel("Weight in %")
plt.title("Optimal Weight of Positions in Millenium ESG Fund")
plt.show()