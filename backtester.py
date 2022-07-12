import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from functools import reduce
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output




class BackTester():
    def __init__(self,
                 file_name,
                 start_date,
                 end_date,
                 gex_bins,
                 starting_capital,
                 transaction_cost,
                 leverage
                 ):
        self.start_date = start_date
        self.end_date = end_date
        self.gex_bins = gex_bins
        self.starting_capital = starting_capital
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        if not os.path.isdir(file_name):
            data_list = [pd.read_csv(f'data/{file}') for file in os.listdir('data') if 'minute' in file]
            price_data = pd.concat(data_list).set_index('trade_date')
            price_data.index = pd.to_datetime(price_data.index)
            gex_data = pd.read_csv('data/spx-gamma-dix.csv').set_index('date')
            gex_data.index = pd.to_datetime(gex_data.index)
            self.data = price_data.join(gex_data[['gex']])
        else:
            self.data = pd.read_csv(file_name, index_col=[0]).set_index('date').loc[start_date:end_date]
        self.data['gex_bin'] = pd.qcut(self.data['gex'], self.gex_bins)

        self.bin_map = {}
        i = 1
        for cut in sorted(pd.qcut(self.data['gex'], self.gex_bins).unique()):
            self.bin_map[cut] = i
            i += 1
        self.data = self.data.loc[start_date:end_date]

    def get_gex_bin_multipliers(self, day):

        gex_bin = self.bin_map[self.data.loc[day]['gex_bin'][0]]

        if gex_bin == 1:
            mr_multiplier = 0
            mom_multiplier = 2
        if gex_bin == self.gex_bins:
            mr_multiplier = 2
            mom_multiplier = 0

        else:
            if gex_bin < self.gex_bins // 2:
                mr_multiplier = 1 - gex_bin / self.gex_bins
                mom_multiplier = 1 + (self.gex_bins - gex_bin) / self.gex_bins
            if gex_bin == self.gex_bins // 2:
                mr_multiplier = 1
                mom_multiplier = 1
            else:
                mr_multiplier = 1 + (self.gex_bins - gex_bin) / self.gex_bins
                mom_multiplier = 1 - gex_bin / self.gex_bins
        multiplier_sum = mr_multiplier + mom_multiplier
        return mr_multiplier/multiplier_sum, mom_multiplier/multiplier_sum

    def evaluate_strategy_for_day(self, day, equity,mr_sma=20,
                                  mr_threshold=0.001,
                                  mr_safety=0.01,
                                  mom_period=20, ):

        mr_multiplier, mom_multiplier = self.get_gex_bin_multipliers(day)

        mean_reversion_pnl, mean_reversion_trades = self.mean_reversion_strategy(data=self.data.loc[day],
                                                          sma=mr_sma,
                                                          safety_threshold=mr_safety,
                                                          threshold=mr_threshold)

        momentum_pnl, momentum_trades = self.momentum_strategy(data=self.data.loc[day],
                                              period=math.floor(mom_period))

        total_transactions = momentum_trades + mean_reversion_trades
        net_return = (((mean_reversion_pnl * mr_multiplier) + (momentum_pnl * mom_multiplier)) * self.leverage) - ((total_transactions)*self.transaction_cost)/equity
        print(f'{day} equity: {self.equity_curve[-1]} return: {net_return}, total transactions: {total_transactions}')
        clear_output(wait = True)
        return net_return

    def evaluate_strategy(self, mr_sma=20,
                          mr_threshold=0.05,
                          mr_safety=0.01,
                          mom_period=20):

        self.strategy_returns = []
        self.benchmark_returns = []
        self.equity_curve = [self.starting_capital]
        for day in self.data.index.unique():
            strategy_return = self.evaluate_strategy_for_day(day,equity = self.equity_curve[-1],
                                                            mr_sma=mr_sma,
                                                            mr_threshold=mr_threshold,
                                                            mr_safety=mr_safety,
                                                            mom_period=mom_period)
            self.strategy_returns.append(strategy_return)
            self.equity_curve.append(self.equity_curve[-1] * (1 + strategy_return))

    def momentum_strategy(self, data, period=1, shorts=True):

        data['Close'] = pd.Series((data['close_ask_price'] + data['close_bid_price']) / 2)
        data['SMA'] = data['Close'].rolling(period).mean()
        returns = data['Close'] / data['Close'].shift(1) - 1
        if shorts:
            position = (data['Close'] - data['SMA']).map(
                lambda x: -1 if x <= 0 else 1)
        else:
            position = returns.rolling(period).mean().map(
                lambda x: 0 if x <= 0 else 1)

        trades = pd.Series(position.diff()).map(lambda x: 1 if x!=0 else 0).sum()
        performance = position.shift(1) * returns
        return performance.cumsum()[-1], trades

    def mean_reversion_strategy(self, data, sma=20, threshold=0.001, safety_threshold=0.01, shorts=True):
        open_price = (data.iloc[0]['close_ask_price'] + data.iloc[0]['close_bid_price']) / 2
        data['Close'] = pd.Series((data['close_ask_price'] + data['close_bid_price']) / 2)
        #data['SMA'] = data['Close'].rolling(sma).mean()
        #data['sigma_up'] = data['SMA'] + data['Close'].rolling(sma).std()
        #data['sigma_down'] = data['SMA'] - data['Close'].rolling(sma).std()
        data['extension'] = (data['Close'] - open_price) / open_price

        data['position'] = np.nan
        data['position'] = np.where(data['extension'] < -threshold,
                                    1, data['position'])
        if shorts:
            data['position'] = np.where(
                data['extension'] > threshold, -1, data['position'])

        data['position'] = np.where(np.abs(data['extension']) < threshold,
                                    0, data['position'])
        data['position'] = data['position'].ffill().fillna(0)

        trades = pd.Series(data['position'].diff()).map(lambda x: 1 if x!=0 else 0).sum()

        # Calculate returns and statistics
        data['returns'] = data['Close'] / data['Close'].shift(1)
        data['log_returns'] = np.log(data['returns'])
        data['strat_returns'] = data['position'].shift(1) * \
                                data['returns']
        data['strat_log_returns'] = data['position'].shift(1) * \
                                    data['log_returns']
        data['cum_returns'] = np.exp(data['log_returns'].cumsum())
        data['strat_cum_returns'] = np.exp(data['strat_log_returns'].cumsum())
        data['peak'] = data['cum_returns'].cummax()
        data['strat_peak'] = data['strat_cum_returns'].cummax()

        ret = data['strat_cum_returns'][-1] - 1

        return ret, trades

    @property
    def equity_curve_strategy(self):
        equity = [self.starting_capital]
        for i in range(len(self.strategy_returns)):
            equity.append(equity[-1] * (1 + self.strategy_returns[i] * self.leverage))
        return equity


    def evaluate_benchmark_for_day(self,day):
        ret = self.benchmark_changes.loc[day][0]
        if math.isnan(ret):
            return 0
        else:
            return ret

    @staticmethod
    def equity_curve_benchmark(starting_capital, leverage, benchmark_returns):
        equity = [starting_capital]
        for i in range(len(benchmark_returns)):
            equity.append(equity[-1] * (1 + benchmark_returns[i] * leverage))
        return equity

if __name__ == '__main__':
    bt = BackTester(file_name = 'combined_minute.csv',
                     start_date='2011-05-02',
                     end_date ='2022-02-18',
                     gex_bins = 16,
                     transaction_cost = 5,
                     starting_capital = 1000000,
                     leverage = 1.2)

    bt.benchmark_changes = bt.data.groupby(level=0).nth(-1)[['close_ask_price']].pct_change()
    rets = [bt.evaluate_benchmark_for_day(day) for day in bt.data.index.unique()]

    bt.evaluate_strategy(mr_sma=20,
                               mr_threshold=0.001,
                               mom_period=20)

