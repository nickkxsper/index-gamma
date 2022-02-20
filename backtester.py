import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class BackTester():
    def __init__(self,
                 file_name,
                 start_date,
                 end_date,
                 gex_bins,
                 starting_capital,
                 leverage
                 ):
        self.start_date = start_date
        self.end_date = end_date
        self.gex_bins = gex_bins
        self.starting_capital = starting_capital
        self.leverage = leverage
        self.data = pd.read_csv(file_name, index_col=[0]).set_index('date').loc[start_date:end_date]
        self.data['gex_bin'] = pd.qcut(self.data['gex'], self.gex_bins)

        self.bin_map = {}
        i = 1
        for cut in sorted(pd.qcut(self.data['gex'], self.gex_bins).unique()):
            self.bin_map[cut] = i
            i += 1

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

        return mr_multiplier, mom_multiplier

    def evaluate_strategy_for_day(self, day, mr_sma=20,
                                  mr_threshold=0.001,
                                  mr_safety=0.01,
                                  mom_period=20, ):

        mean_reversion_pnl = self.mean_reversion_strategy(data=self.data.loc[day],
                                                          sma=mr_sma,
                                                          safety_threshold=mr_safety,
                                                          threshold=mr_threshold)

        momentum_pnl = self.momentum_strategy(data=self.data.loc[day],
                                              period=math.floor(mom_period))

        mr_multiplier, mom_multiplier = self.get_gex_bin_multipliers(day)

        return (mean_reversion_pnl * mr_multiplier) + (momentum_pnl * mom_multiplier)

    def evaluate_strategy(self, mr_sma=20,
                          mr_threshold=0.05,
                          mr_safety=0.01,
                          mom_period=20):

        self.strategy_returns = []
        self.benchmark_returns = []
        for day in self.data.index.unique():
            self.strategy_returns.append(self.evaluate_strategy_for_day(day,
                                                                        mr_sma=mr_sma,
                                                                        mr_threshold=mr_threshold,
                                                                        mr_safety=mr_safety,
                                                                        mom_period=mom_period))

    def momentum_strategy(self, data, period=1, shorts=True):

        data['Close'] = pd.Series((data['close_ask_price'] + data['close_bid_price']) / 2)
        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
        if shorts:
            position = returns.rolling(period).mean().map(
                lambda x: -1 if x <= 0 else 1)
        else:
            position = returns.rolling(period).mean().map(
                lambda x: 0 if x <= 0 else 1)
        performance = position.shift(1) * returns
        return performance.cumsum()[-1]

    def mean_reversion_strategy(self, data, sma=20, threshold=0.001, safety_threshold=0.01, shorts=True):
        data['Close'] = pd.Series((data['close_ask_price'] + data['close_bid_price']) / 2)
        data['SMA'] = data['Close'].rolling(sma).mean()
        data['sigma_up'] = data['SMA'] + data['Close'].rolling(sma).std()
        data['sigma_down'] = data['SMA'] - data['Close'].rolling(sma).std()
        data['extension'] = (data['Close'] - data['SMA']) / data['SMA']

        data['position'] = np.nan
        data['position'] = np.where(data['extension'] < -threshold,
                                    1, data['position'])
        if shorts:
            data['position'] = np.where(
                data['extension'] > threshold, -1, data['position'])

        data['position'] = np.where(np.abs(data['extension']) < threshold,
                                    0, data['position'])
        data['position'] = data['position'].ffill().fillna(0)

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

        return ret

    @property
    def equity_curve_strategy(self):
        equity = [self.starting_capital]
        for i in range(len(self.strategy_returns)):
            equity.append(equity[-1] * (1 + self.strategy_returns[i] * self.leverage))
        return equity




