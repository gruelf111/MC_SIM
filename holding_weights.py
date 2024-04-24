import matplotlib.pyplot as plt
import numpy as np

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

weights = list(np.array(list(stocklist.values()))*100)
tickers = list(stocklist.keys())
weights_sum = np.array(list(stocklist.values())).sum()
cash_weight = round(1 - weights_sum,2)*100

plt.barh(tickers, weights, color='skyblue')
plt.barh("CASH",cash_weight, color='grey')
for index, value in enumerate(weights):
    plt.text(value, index, f"{value:.1f}")
plt.text(cash_weight, len(weights), f"{cash_weight:.1f}")
plt.ylabel("Yahoo Finance Ticker")
plt.xlabel("Weight in %")
plt.title("Weight of Positions in Millenium ESG Fund")
plt.show()