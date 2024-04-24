import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

# get data
def get_data(stocks, start, end):
    stockData = pd.DataFrame()
    for stock in stocks:
        try:
            data = pdr.get_data_yahoo(stock, start, end)['Close']
            stockData[stock] = data
        except Exception as e:
            print(f"Error retrieving data for {stock}: {e}")
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stocklist = {'MSFT':0.092, 
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
             'SUSW.L':0.099
             }
stocks = list(stocklist.keys())
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

print(meanReturns)

weights = np.array(list(stocklist.values()))

mc_sims = 100
T = 100

meanM = np.full(shape=(T, len(stocks)), fill_value= meanReturns)
meanM = meanM.T
print(meanM)
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
intitial_portfolio_value = 1000000

for m in range (0,mc_sims):
    Z = np.random.normal(size=(T, len(stocks)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L,Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)* intitial_portfolio_value

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value')
plt.xlabel('Days')
plt.title('MC Simulation of Millenium ESG Fund')
plt.show()