import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

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

dailyReturns = log_returns.dot(weight)

expected_return = dailyReturns.mean() * 252
vola = dailyReturns.std() * np.sqrt(252)
skew = dailyReturns.skew()
sharpe = expected_return / vola

metrics = {'Expected Return': [expected_return],
           'Volatility': [vola],
            'Skewness': [skew],
             'Sharpe Ratio': [sharpe]}
df_metrics = pd.DataFrame(metrics)

print(df_metrics)
