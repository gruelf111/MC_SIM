import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

# Your existing code to fetch data and calculate dailyReturns
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
data = yf.download(stocks, start=startdate, end=enddate)['Close'][stocks]

df = pd.DataFrame(data=data)
df = df.dropna()

log_returns = np.log(df) - np.log(df.shift(1))
weights = np.array(list(stocklist.values()))

dailyReturns = log_returns.dot(weights)

# Calculate cumulative returns
cumulative_returns = (1 + dailyReturns).cumprod()

# Calculate the running maximum
running_max = cumulative_returns.cummax()

# Calculate drawdowns
drawdowns = running_max - cumulative_returns

# Calculate maximum drawdown
max_drawdown = drawdowns.max()

print(f"Maximum Drawdown: {max_drawdown}")

# Plotting the cumulative returns and drawdowns
plt.figure(figsize=(14, 7))
plt.subplot(2, 1, 1)
plt.plot(cumulative_returns, label='Cumulative Returns')
plt.plot(running_max, label='Running Maximum', linestyle='--')
plt.title('Cumulative Returns and Running Maximum')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(drawdowns, label='Drawdown', color='red')
plt.fill_between(drawdowns.index, drawdowns, color='red', alpha=0.3)
plt.title('Drawdown Over Time')
plt.legend()

plt.tight_layout()
plt.show()